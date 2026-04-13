"""
app.py  –  MindTrace · Emotion Prediction Flask API
=====================================================
Author : Aye Khin Khin Hpone (Yolanda Lim) · ST125970
         Computer Science, Asian Institute of Technology

Thesis : MindTrace — Text Mining and NLP-Driven Emotion Prediction
         Using Machine Learning and Deep Learning Approaches

Research Question:
  How do traditional ML models (SVM, XGBoost) compare with deep learning
  models (CNN, BiLSTM) in multi-class emotion classification from social
  media text, when evaluated under a unified preprocessing pipeline,
  standardised dataset partitioning, and consistent evaluation metrics?

Endpoints
---------
GET  /          → MindTrace web UI
POST /predict   → { "text": "..." } → emotion + probabilities + NLP steps
GET  /stats     → model performance + class distribution (for dashboard)
GET  /health    → health check
"""

import os, re

import joblib, emoji, nltk, numpy as np
from flask import Flask, request, jsonify, render_template

for res in ["stopwords", "wordnet", "omw-1.4"]:
    try: nltk.data.find(f"corpora/{res}")
    except LookupError: nltk.download(res, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Emotion classes — label map from dataset ─────────────────────────────────
# 0=Sadness, 1=Joy, 2=Love, 3=Anger, 4=Fear, 5=Surprise
LABEL_MAP = {0:"Sadness", 1:"Joy", 2:"Love", 3:"Anger", 4:"Fear", 5:"Surprise"}

# Colors MUST match JS EMOTION_META in index.html exactly
EMOTION_META = {
    "Sadness":  {"emoji":"😢", "color":"#14b8a6", "orb":"rgba(20,184,166,0.15)",
                 "description":"A feeling of sorrow or deep unhappiness"},
    "Joy":      {"emoji":"😊", "color":"#0fffa0", "orb":"rgba(15,255,160,0.18)",
                 "description":"A feeling of great pleasure or happiness"},
    "Love":     {"emoji":"❤️", "color":"#06b6d4", "orb":"rgba(6,182,212,0.15)",
                 "description":"A feeling of deep affection and warmth"},
    "Anger":    {"emoji":"😠", "color":"#f87171", "orb":"rgba(248,113,113,0.15)",
                 "description":"A strong feeling of displeasure or hostility"},
    "Fear":     {"emoji":"😨", "color":"#a78bfa", "orb":"rgba(167,139,250,0.15)",
                 "description":"An unpleasant emotion caused by perceived danger or threat"},
    "Surprise": {"emoji":"😲", "color":"#fbbf24", "orb":"rgba(251,191,36,0.15)",
                 "description":"A feeling of mild astonishment or unexpected discovery"},
}

# ── Negation words preserved — Section 3.3 of MindTrace thesis ───────────────
# These carry emotional polarity and must NOT be removed as stopwords
NEGATION_WORDS = {
    "not","never","no","nor","neither","nothing","nobody",
    "nowhere","without","very","extremely","barely","hardly"
}

STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS
LEMMATIZER = WordNetLemmatizer()

# Chat words / informal abbreviations — standardised per Section 3.3
CHAT_WORDS = {
    "u":"you","r":"are","ur":"your","lol":"laugh out loud","omg":"oh my god",
    "brb":"be right back","btw":"by the way","idk":"i do not know",
    "imo":"in my opinion","tbh":"to be honest","ngl":"not gonna lie",
    "smh":"shaking my head","ikr":"i know right","nvm":"never mind",
    "gonna":"going to","wanna":"want to","gotta":"got to","kinda":"kind of",
    "cuz":"because","bc":"because","thx":"thanks","ty":"thank you",
    "np":"no problem","asap":"as soon as possible","irl":"in real life",
    "dm":"direct message","gr8":"great","luv":"love","plz":"please",
    "pls":"please","rn":"right now",
}

# ── Model results — Table 9, MindTrace thesis ────────────────────────────────
# Actual metrics from Emotion_prediction_source_code.ipynb on balanced 89,832-sample corpus
MODEL_STATS = {
    "BiLSTM":  {"train":94.7,"val":94.2,"test":93.9,"precision":94.1,"recall":93.9,"f1":93.9},
    "CNN":     {"train":93.2,"val":92.8,"test":92.5,"precision":92.7,"recall":92.5,"f1":92.5},
    "SVM":     {"train":94.4,"val":92.1,"test":91.8,"precision":92.0,"recall":91.8,"f1":91.8},
    "XGBoost": {"train":93.3,"val":91.1,"test":90.8,"precision":91.0,"recall":90.8,"f1":90.7},
}

# ── Class distribution — Table 4, MindTrace thesis ──────────────────────────
CLASS_DISTRIBUTION = {
    "Joy":     {"count":141067,"pct":33.84,"note":"Largest class — risk of bias if unweighted"},
    "Sadness": {"count":121187,"pct":29.07,"note":"Second largest; close to Fear/Love semantically"},
    "Anger":   {"count":57317, "pct":13.75,"note":"Mid-tier; distinct vocabulary aids separation"},
    "Fear":    {"count":47712, "pct":11.45,"note":"Often confused with Surprise due to overlap"},
    "Love":    {"count":34554, "pct":8.29, "note":"Minority class; requires balancing strategy"},
    "Surprise":{"count":14972, "pct":3.59, "note":"Smallest — highest misclassification risk"},
}

# ── NLP Preprocessing pipeline — Section 3.3 ─────────────────────────────────
# This function MUST mirror train_pipeline.py exactly — any divergence = data leakage
def clean_text(text: str) -> dict:
    """
    Full NLP pipeline per MindTrace thesis Section 3.3.
    Returns both the cleaned text AND a trace of each step for the UI.
    """
    steps = {}
    original = str(text)

    # Step 1: Lowercase
    t = original.lower()
    steps["lowercased"] = t

    # Step 2: Whitespace stripping
    t = re.sub(r"\s+", " ", t).strip()
    steps["whitespace_stripped"] = t

    # Step 3: URL removal
    t = re.sub(r"http\S+|www\S+", "", t)
    steps["url_removed"] = t

    # Step 4: Emoji removal
    t = emoji.replace_emoji(t, replace="")
    steps["emoji_removed"] = t

    # Step 5: Special char removal (keep a-z and spaces)
    t = re.sub(r"[^a-z\s]", "", t)
    steps["special_removed"] = t

    # Step 6: Chat word expansion
    words = t.split()
    expanded = [CHAT_WORDS.get(w, w) for w in words]
    steps["chat_expanded"] = " ".join(expanded)

    # Step 7: Stopword removal (negations preserved)
    filtered = [w for w in expanded if w not in STOP_WORDS]
    negations_kept = [w for w in filtered if w in NEGATION_WORDS]
    steps["stopwords_removed"] = " ".join(filtered)
    steps["negations_preserved"] = negations_kept

    # Step 8: Lemmatisation
    lemmatised = [LEMMATIZER.lemmatize(w) for w in filtered]
    steps["lemmatised"] = lemmatised
    final = " ".join(lemmatised)
    steps["final"] = final

    return {"cleaned": final, "steps": steps, "tokens": lemmatised}

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
if os.path.exists(MODEL_PATH):
    print(f"[MindTrace] Loading {MODEL_PATH} …", flush=True)
    pipeline_model = joblib.load(MODEL_PATH)
    print("[MindTrace] Ready ✓", flush=True)
else:
    pipeline_model = None
    print(f"[MindTrace] Warning: {MODEL_PATH} not found — prediction disabled", flush=True)

prediction_history = []
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if pipeline_model is None:
        return jsonify({"error": "Model not loaded. Run train_pipeline.py first."}), 503
    body = request.get_json(force=True, silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please provide a non-empty 'text' field."}), 400
    if len(text) > 1000:
        return jsonify({"error": "Text must be 1 000 characters or fewer."}), 400

    nlp = clean_text(text)
    cleaned = nlp["cleaned"]
    if not cleaned:
        return jsonify({"error": "Text becomes empty after preprocessing. Try a longer or different input."}), 400

    probs   = pipeline_model.predict_proba([cleaned])[0]
    pred_id = int(np.argmax(probs))
    label   = LABEL_MAP[pred_id]
    meta    = EMOTION_META[label]

    all_emotions = sorted([
        {
            "label":       LABEL_MAP[i],
            "probability": round(float(probs[i]) * 100, 2),
            "emoji":       EMOTION_META[LABEL_MAP[i]]["emoji"],
            "color":       EMOTION_META[LABEL_MAP[i]]["color"],
        }
        for i in range(6)
    ], key=lambda x: x["probability"], reverse=True)

    entry = {
        "text":       text[:90] + ("…" if len(text) > 90 else ""),
        "prediction": label,
        "confidence": round(float(probs[pred_id]) * 100, 2),
        "emoji":      meta["emoji"],
        "color":      meta["color"],
    }
    prediction_history.insert(0, entry)
    if len(prediction_history) > 50: prediction_history.pop()

    return jsonify({
        "prediction":   label,
        "emoji":        meta["emoji"],
        "color":        meta["color"],
        "orb":          meta["orb"],
        "description":  meta["description"],
        "confidence":   round(float(probs[pred_id]) * 100, 2),
        "all_emotions": all_emotions,
        "cleaned_text": cleaned,
        "tokens":       nlp["tokens"][:40],
        "nlp_steps":    nlp["steps"],
        "negations_kept": nlp["steps"].get("negations_preserved", []),
        "history":      prediction_history[:8],
    })

@app.route("/stats")
def stats():
    return jsonify({
        "model_stats":        MODEL_STATS,
        "class_distribution": CLASS_DISTRIBUTION,
        "dataset_size":       416809,
        "num_classes":        6,
        "deployed_model":     "SVM + TF-IDF Pipeline (sklearn)",
        "best_model":         "BiLSTM (93.9% test accuracy)",
        "research_question":  (
            "How do traditional ML models (SVM, XGBoost) compare with deep learning "
            "models (CNN, BiLSTM) in multi-class emotion classification from social "
            "media text, when evaluated under a unified preprocessing pipeline, "
            "standardised dataset partitioning, and consistent evaluation metrics?"
        ),
        "imbalance_ratio":    "9.5:1 (Joy:Surprise)",
        "hard_pairs":         ["Fear / Surprise", "Joy / Love"],
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH, "project": "MindTrace"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
