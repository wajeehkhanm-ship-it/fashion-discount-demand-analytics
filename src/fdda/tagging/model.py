"""Train and evaluate per-field product taggers (TF-IDF -> XGBoost / logistic regression).

For each field three models are compared on the validation styles:
* most_frequent - always predicts the most common training label (the baseline to beat)
* logreg        - multinomial logistic regression on TF-IDF features
* xgboost       - gradient-boosted trees on the same features, early-stopped on validation

The model with the best validation accuracy is kept; all three are scored on the test styles.
Each prediction carries a confidence (the top class probability), so suggestions can be
auto-accepted above a threshold and only the rest reviewed by a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from fdda.log import get_logger

log = get_logger(__name__)
MODELS = ["most_frequent", "logreg", "xgboost"]


@dataclass
class FieldTagger:
    """A fitted classifier for one field, with its label encoder."""

    name: str
    kind: str
    encoder: LabelEncoder
    model: Any = None
    majority: int | None = None
    selector: Any = None  # optional feature selection applied before the model

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities (n_samples x n_classes)."""
        if self.kind == "most_frequent":
            p = np.zeros((X.shape[0], len(self.encoder.classes_)))
            p[:, self.majority] = 1.0
            return p
        if self.selector is not None:
            X = self.selector.transform(X)
        return self.model.predict_proba(X)

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(labels, confidence) for each row."""
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return self.encoder.inverse_transform(idx), proba.max(axis=1)


@dataclass
class TaggerBundle:
    """Everything needed to tag new products: the vectorizer and one tagger per field."""

    vectorizer: TfidfVectorizer
    taggers: dict[str, FieldTagger] = field(default_factory=dict)

    def predict(self, texts: list[str] | pd.Series) -> pd.DataFrame:
        """Suggested label and confidence per field for each text."""
        X = self.vectorizer.transform(list(texts))
        out = {}
        for name, t in self.taggers.items():
            labels, conf = t.predict(X)
            out[name], out[f"{name}_confidence"] = labels, conf
        return pd.DataFrame(out)


def fit_vectorizer(texts: pd.Series, cfg: dict[str, Any]) -> TfidfVectorizer:
    """TF-IDF over word unigrams/bigrams, fit on training texts only."""
    vec = TfidfVectorizer(
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=cfg["min_df"],
        max_features=cfg["max_features"],
        sublinear_tf=cfg["sublinear_tf"],
        lowercase=True,
        strip_accents="unicode",
    )
    vec.fit(texts)
    return vec


def _encode(enc: LabelEncoder, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Encoded labels and a mask of rows whose label was seen in training."""
    known = y.isin(enc.classes_).to_numpy()
    codes = np.full(len(y), -1)
    codes[known] = enc.transform(y[known])
    return codes, known


def train_field(
    name: str,
    Xtr,
    ytr: pd.Series,
    Xva,
    yva: pd.Series,
    kind: str,
    cfg: dict[str, Any],
    seed: int,
) -> FieldTagger:
    """Fit one model kind for one field."""
    enc = LabelEncoder().fit(ytr)
    y = enc.transform(ytr)
    if kind == "most_frequent" or len(enc.classes_) < 2:  # one class: nothing to learn
        return FieldTagger(name, "most_frequent", enc, majority=int(np.bincount(y).argmax()))
    if kind == "logreg":
        m = LogisticRegression(C=cfg["logreg"]["C"], max_iter=cfg["logreg"]["max_iter"])
        m.fit(Xtr, y)
        return FieldTagger(name, kind, enc, model=m)
    if kind == "xgboost":
        from sklearn.feature_selection import SelectKBest, chi2
        from xgboost import XGBClassifier

        xc = dict(cfg["xgboost"])
        stop = xc.pop("early_stopping_rounds")
        k = xc.pop("max_features_selected", None)
        selector = None
        if k and k < Xtr.shape[1]:  # trees on 30k sparse text features are slow; keep the k best
            selector = SelectKBest(chi2, k=k).fit(Xtr, y)
            Xtr, Xva = selector.transform(Xtr), selector.transform(Xva)
        yv, known = _encode(enc, yva)
        params = {
            **xc,
            "early_stopping_rounds": stop,
            "eval_metric": "mlogloss",
            "random_state": seed,
            "n_jobs": -1,
        }
        if len(enc.classes_) == 2:
            params.update(objective="binary:logistic", eval_metric="logloss")
        else:
            params.update(objective="multi:softprob", num_class=len(enc.classes_))
        m = XGBClassifier(**params)
        m.fit(Xtr, y, eval_set=[(Xva[known], yv[known])], verbose=False)
        return FieldTagger(name, kind, enc, model=m, selector=selector)
    raise ValueError(f"unknown model kind: {kind}")


def evaluate(
    y_true: pd.Series, y_pred: np.ndarray, conf: np.ndarray, thresholds: list[float]
) -> dict:
    """Accuracy, macro/weighted F1 and a confidence-threshold table."""
    y_true = np.asarray(y_true)
    correct = y_pred == y_true
    table = []
    for t in thresholds:
        sel = conf >= t
        table.append(
            {
                "threshold": t,
                "coverage": float(sel.mean()),
                "accuracy_above": float(correct[sel].mean()) if sel.any() else float("nan"),
                "accuracy_below": float(correct[~sel].mean()) if (~sel).any() else float("nan"),
            }
        )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n": int(len(y_true)),
        "thresholds": table,
    }


def per_class_report(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Precision / recall / F1 / support per class."""
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = [{"label": k, **v} for k, v in rep.items() if isinstance(v, dict)]
    return pd.DataFrame(rows).rename(columns={"f1-score": "f1"})


def train_all(
    df: pd.DataFrame, cfg: dict[str, Any], seed: int
) -> tuple[TaggerBundle, pd.DataFrame, dict[str, Any]]:
    """Fit all model kinds for every field, keep the best per field by validation accuracy.

    `df` needs text, split (train/val/test) and one column per field. Returns the bundle of
    chosen models, a long metrics table (field x model x split) and evaluation details
    (threshold tables, per-class reports, test predictions, title-only test accuracy).
    """
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    vec = fit_vectorizer(tr["text"], cfg["tfidf"])
    Xtr, Xva, Xte = (vec.transform(d["text"]) for d in (tr, va, te))
    Xte_title = vec.transform(te["title"].fillna(""))
    bundle = TaggerBundle(vec)
    rows, details = [], {}
    thresholds = cfg["confidence_thresholds"]
    for f in cfg["fields"]:
        best, best_acc, fitted = None, -1.0, {}
        for kind in cfg.get("models", MODELS):
            t = train_field(f, Xtr, tr[f], Xva, va[f], kind, cfg, seed)
            fitted[kind] = t
            for split, X, d in (("val", Xva, va), ("test", Xte, te)):
                pred, conf = t.predict(X)
                ev = evaluate(d[f], pred, conf, thresholds)
                rows.append(
                    {
                        "field": f,
                        "model": kind,
                        "split": split,
                        **{k: v for k, v in ev.items() if k != "thresholds"},
                    }
                )
                if split == "val" and ev["accuracy"] > best_acc:
                    best, best_acc = kind, ev["accuracy"]
            log.info("%-15s %-13s val acc %.3f", f, kind, rows[-2]["accuracy"])
        chosen = fitted[best]
        bundle.taggers[f] = chosen
        pred, conf = chosen.predict(Xte)
        tpred, tconf = chosen.predict(Xte_title)
        details[f] = {
            "chosen_model": best,
            "test": evaluate(te[f], pred, conf, thresholds),
            "test_title_only": evaluate(te[f], tpred, tconf, thresholds),
            "per_class": per_class_report(te[f], pred),
            "predictions": pd.DataFrame(
                {
                    "id": te["id"].to_numpy(),
                    "true": te[f].to_numpy(),
                    "pred": pred,
                    "confidence": conf,
                }
            ),
        }
    return bundle, pd.DataFrame(rows), details


def auto_accept_summary(
    details: dict[str, Any], fields: list[str], threshold: float
) -> dict[str, float]:
    """Share of test products whose suggestions for *all* fields clear `threshold`, and how
    accurate those fully auto-accepted products are (every field correct)."""
    preds = [details[f]["predictions"].set_index("id") for f in fields]
    conf = pd.concat([p["confidence"] for p in preds], axis=1).min(axis=1)
    correct = pd.concat([(p["pred"] == p["true"]) for p in preds], axis=1).all(axis=1)
    sel = conf >= threshold
    return {
        "threshold": threshold,
        "products": int(len(conf)),
        "auto_accept_share": float(sel.mean()),
        "all_fields_correct_when_auto_accepted": float(correct[sel].mean())
        if sel.any()
        else float("nan"),
        "all_fields_correct_overall": float(correct.mean()),
    }
