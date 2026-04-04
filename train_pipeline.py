"""
train_pipeline.py  –  MindTrace Training Script
================================================
Author  : Aye Khin Khin Hpone (Yolanda Lim) · ST125970
          Computer Science, Asian Institute of Technology

MindTrace — Text Mining and NLP-Driven Emotion Prediction
          Using Machine Learning and Deep Learning Approaches

Purpose : Build and serialise the sklearn Pipeline (TF-IDF → SVM) that
          app.py loads at startup. Run this ONCE before deploying.

Usage:
    # default — looks for data/text.xlsx
    python train_pipeline.py

    # custom data path
    python train_pipeline.py --data /path/to/text.xlsx

    # custom output path
    python train_pipeline.py --data data/text.xlsx --model-out model.pkl

The produced model.pkl is a full sklearn Pipeline:
    TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    └── SVC(C=1.0, kernel='linear', probability=True)

app.py just calls:
    pipeline.predict_proba([cleaned_text])
No manual vectorising or encoding at inference time.

NLP preprocessing pipeline mirrors app.py clean_text() exactly.
Any divergence between the two = training/serving skew = wrong predictions.
"""

import argparse
import os
import re
import sys
import subprocess

# ── auto-install ──────────────────────────────────────────────────────────────
for pkg in ["openpyxl", "emoji", "nltk", "scikit-learn", "joblib", "pandas", "numpy"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import pandas as pd
import numpy as np
import joblib
import nltk
import emoji

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# ── NLTK resources ────────────────────────────────────────────────────────────
for res in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{res}")
    except LookupError:
        nltk.download(res, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── LABEL MAP — must match app.py ─────────────────────────────────────────────
# Dataset encoding: 0=Sadness, 1=Joy, 2=Love, 3=Anger, 4=Fear, 5=Surprise
LABEL_MAP = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}

# ── NEGATION WORDS — preserved from stopword removal (Section 3.4) ────────────
# !! MUST be identical to the set in app.py !!
NEGATION_WORDS = {
    "not", "never", "no", "nor", "neither", "nothing", "nobody",
    "nowhere", "without", "very", "extremely", "barely", "hardly"
}

STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()

# ── CHAT WORDS — must match app.py ────────────────────────────────────────────
CHAT_WORDS = {
    "u": "you", "r": "are", "ur": "your", "lol": "laugh out loud",
    "omg": "oh my god", "brb": "be right back", "btw": "by the way",
    "idk": "i do not know", "imo": "in my opinion", "tbh": "to be honest",
    "ngl": "not gonna lie", "smh": "shaking my head", "ikr": "i know right",
    "nvm": "never mind", "gonna": "going to", "wanna": "want to",
    "gotta": "got to", "kinda": "kind of", "cuz": "because",
    "bc": "because", "thx": "thanks", "ty": "thank you",
    "np": "no problem", "asap": "as soon as possible", "irl": "in real life",
    "dm": "direct message", "gr8": "great", "luv": "love",
    "plz": "please", "pls": "please", "rn": "right now",
}


def clean_text(text: str) -> str:
    """
    NLP preprocessing pipeline — Section 3.4 of MindTrace thesis.

    Steps (must mirror app.py clean_text exactly):
      1. Lowercase
      2. URL removal
      3. Emoji removal
      4. Special character removal (keep a-z and spaces)
      5. Chat word expansion
      6. Stopword removal (negation words preserved)
      7. Lemmatisation (NLTK WordNetLemmatizer)
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [CHAT_WORDS.get(w, w) for w in words]
    words = [w for w in words if w not in STOP_WORDS]
    words = [LEMMATIZER.lemmatize(w) for w in words]
    return " ".join(words)


def load_and_balance(path: str, seed: int = 42) -> pd.DataFrame:
    """Load dataset and downsample to the smallest class size."""
    print(f"[1/5] Loading dataset from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            f"Download from: https://www.kaggle.com/datasets/nelgiriyewithana/emotions\n"
            f"Then place it at: {path}"
        )

    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .xlsx or .csv")

    # Keep only needed columns
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    # Validate label range
    assert set(df["label"].unique()).issubset({0,1,2,3,4,5}), \
        "Unexpected label values — expected 0–5"

    # Show raw distribution
    print("      Raw class distribution:")
    for label_id, count in sorted(df["label"].value_counts().items()):
        print(f"        {LABEL_MAP[label_id]:<10} {count:>7,}")

    # Downsample to minority class (Surprise ≈ 14,872)
    min_count = df["label"].value_counts().min()
    print(f"\n      Balancing: downsampling all classes to {min_count:,} samples")
    print(f"      (Original imbalance ratio ~9.5:1 → equalised)")
    balanced = (
        df.groupby("label", group_keys=False)
          .apply(lambda x: x.sample(min_count, random_state=seed))
          .reset_index(drop=True)
    )
    print(f"      Balanced total: {len(balanced):,} samples across 6 classes")
    return balanced


def main():
    parser = argparse.ArgumentParser(
        description="MindTrace — Train SVM Pipeline and save model.pkl"
    )
    parser.add_argument(
        "--data", default="data/text.xlsx",
        help="Path to dataset file (.xlsx or .csv)"
    )
    parser.add_argument(
        "--model-out", default="model.pkl",
        help="Output path for serialised pipeline"
    )
    args = parser.parse_args()

    # ── Load & balance ────────────────────────────────────────────────────────
    df = load_and_balance(args.data)

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("\n[2/5] Preprocessing text (NLP pipeline — Section 3.4) …")
    df["clean"] = df["text"].apply(clean_text)
    empty = (df["clean"].str.strip() == "").sum()
    if empty > 0:
        print(f"      Warning: {empty} rows became empty after preprocessing — dropping")
        df = df[df["clean"].str.strip() != ""].reset_index(drop=True)

    # ── Split: 80% train / 20% test; 20% of train = validation ──────────────
    X, y = df["clean"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=0, stratify=y_train
    )
    print(f"      Train: {len(X_tr):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # ── Build Pipeline ────────────────────────────────────────────────────────
    print("\n[3/5] Building sklearn Pipeline: TF-IDF → SVM …")
    print("      TfidfVectorizer: max_features=5000, ngram_range=(1,2)")
    print("      SVC: kernel=linear, probability=True")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("svm",   SVC(C=1.0, kernel="linear", probability=True)),
    ])

    # ── GridSearchCV ──────────────────────────────────────────────────────────
    param_grid = {
        "svm__C":      [0.1, 1],
        "svm__kernel": ["linear"],
    }
    print("\n[4/5] GridSearchCV (cv=3, scoring=accuracy) …")
    grid = GridSearchCV(
        pipeline, param_grid,
        cv=3, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    print(f"      Best params: {grid.best_params_}")
    print(f"      Best CV accuracy: {grid.best_score_:.4f}")

    # Validation check
    val_pred = best.predict(X_val)
    val_acc  = accuracy_score(y_val, val_pred)
    val_f1   = f1_score(y_val, val_pred, average="macro")
    print(f"      Validation accuracy: {val_acc:.4f}  |  Macro-F1: {val_f1:.4f}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    y_pred = best.predict(X_test)
    labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec  = recall_score(y_test, y_pred, average="macro")
    f1   = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'='*58}")
    print(f"  MindTrace — SVM Test Results (20% hold-out)")
    print(f"{'='*58}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Macro-F1  : {f1:.4f}  ({f1*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"{'='*58}")
    print(classification_report(y_test, y_pred, target_names=labels))

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"[5/5] Saving pipeline → {args.model_out}")
    joblib.dump(best, args.model_out)
    size_mb = os.path.getsize(args.model_out) / (1024 * 1024)
    print(f"      Saved ({size_mb:.1f} MB)")
    print(f"\n  ✓  Done. Deploy with:")
    print(f"       python app.py")
    print(f"     Or with Docker:")
    print(f"       docker build -t mindtrace . && docker run -p 5000:5000 mindtrace")


if __name__ == "__main__":
    main()
