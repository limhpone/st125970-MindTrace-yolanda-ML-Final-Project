# MindTrace — Text Mining & NLP-Driven Emotion Prediction

**Author:** Aye Khin Khin Hpone (Yolanda Lim) · ST125970  
**Programme:** Computer Science, Asian Institute of Technology  
**Thesis:** *MindTrace: Text Mining and NLP-Driven Emotion Prediction Using Machine Learning and Deep Learning Approaches*

---

## Research Question

> How do traditional ML models (SVM, XGBoost) compare with deep learning models (CNN, BiLSTM) in multi-class emotion classification from social media text, when evaluated under a unified preprocessing pipeline, standardised dataset partitioning, and consistent evaluation metrics?

---

## Model Performance — Table 4.1

| Model | Training | Validation | **Test Acc.** | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **BiLSTM** ✓ best | 96.3% | 94.5% | **95.2%** | 95.4% | 95.2% | 95.2% |
| CNN | 94.2% | 93.2% | 92.6% | 92.8% | 92.6% | 92.5% |
| SVM | 93.5% | 91.3% | 91.3% | 91.5% | 91.4% | 91.3% |
| XGBoost | 92.9% | 90.5% | 90.6% | 90.8% | 90.7% | 90.6% |

> **Deployed model:** SVM + TF-IDF sklearn Pipeline — CPU inference, no GPU required, instant startup

---

## Project Structure

```
mindtrace/
├── app.py                  # Flask API — loads model.pkl, serves UI
├── train_pipeline.py       # Build model.pkl from the dataset
├── test_predict.ipynb      # Inference tests — 6 classes + batch prediction
├── model.pkl               # Serialised sklearn Pipeline (generate with train_pipeline.py)
├── requirements.txt        # Pinned Python dependencies
├── Dockerfile              # Container definition (python:3.11-slim + gunicorn)
├── templates/
│   └── index.html          # MindTrace web UI (4 tabs — self-contained)
└── data/
    └── text.xlsx           # Raw dataset — download separately from Kaggle
```

---

## Dataset — Table 3.3

**Source:** [Kaggle Emotions Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions)  
416,809 labelled Twitter samples · 6 emotion classes · Imbalance ratio 9.5:1

| Label | Code | Count | % | Note |
|---|---|---|---|---|
| Joy | 1 | 141,067 | 33.84% | Largest — risk of bias if unweighted |
| Sadness | 0 | 121,187 | 29.07% | Semantically close to Fear/Love |
| Anger | 3 | 57,317 | 13.75% | Distinct vocabulary aids separation |
| Fear | 4 | 47,712 | 11.45% | Often confused with Surprise |
| Love | 2 | 34,554 | 8.29% | Minority class |
| Surprise | 5 | 14,872 | 3.59% | Smallest — highest misclassification risk |

**High-risk confusion pairs (Section 4.1):** Fear ↔ Surprise · Sadness ↔ Love

---

## NLP Preprocessing Pipeline — Section 3.4

Applied identically in `train_pipeline.py` during training and in `app.py` at inference time. Any divergence between the two causes training/serving skew.

| Step | Operation | Key Detail |
|---|---|---|
| 1 | Lowercase | `str.lower()` |
| 2 | URL removal | `re.sub(r'http\S+\|www\S+', '', t)` |
| 3 | Emoji removal | `emoji.replace_emoji(t, replace='')` |
| 4 | Special character removal | Keep only `[a-z\s]` |
| 5 | Chat word expansion | `u→you`, `lol→laugh out loud`, 28 rules |
| 6 | Stopword removal | NLTK English stopwords — **negation words preserved** |
| 7 | Lemmatisation | `nltk.WordNetLemmatizer` |

**Negation words preserved** (Section 3.4): `not, never, no, nor, neither, nothing, nobody, nowhere, without, very, extremely, barely, hardly` — these carry emotional polarity and their removal would lose discriminative signal.

---

## Quick Start

### Option 1 — Local Python

```bash
# 1. Clone / download this folder
cd mindtrace

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    https://www.kaggle.com/datasets/nelgiriyewithana/emotions
#    Save as: data/text.xlsx

# 4. Train and save model.pkl (~5–15 min depending on hardware)
python train_pipeline.py --data data/text.xlsx

# 5. Start the server
python app.py
# → Open http://localhost:5000
```

### Option 2 — Docker

```bash
# Build (model.pkl must exist first — run step 4 above)
docker build -t mindtrace .

# Run
docker run -p 5000:5000 mindtrace
# → Open http://localhost:5000
```

### Option 3 — Docker Hub (if published)

```bash
docker pull <your-dockerhub-username>/mindtrace
docker run -p 5000:5000 <your-dockerhub-username>/mindtrace
```

---

## API Reference

### `POST /predict`

Accepts raw text, returns emotion label with probabilities and NLP step trace.

**Request**
```json
{ "text": "I feel so happy today!" }
```

**Response**
```json
{
  "prediction":   "Joy",
  "emoji":        "😊",
  "color":        "#0fffa0",
  "orb":          "rgba(15,255,160,0.18)",
  "description":  "A feeling of great pleasure or happiness",
  "confidence":   87.43,
  "all_emotions": [
    { "label": "Joy",     "probability": 87.43, "emoji": "😊", "color": "#0fffa0" },
    { "label": "Love",    "probability":  6.12, "emoji": "❤️", "color": "#06b6d4" },
    { "label": "Sadness", "probability":  3.21, "emoji": "😢", "color": "#14b8a6" },
    { "label": "Surprise","probability":  1.45, "emoji": "😲", "color": "#fbbf24" },
    { "label": "Anger",   "probability":  1.02, "emoji": "😠", "color": "#f87171" },
    { "label": "Fear",    "probability":  0.77, "emoji": "😨", "color": "#a78bfa" }
  ],
  "cleaned_text": "feel happy today",
  "tokens":       ["feel", "happy", "today"],
  "nlp_steps": {
    "lowercased":         "i feel so happy today!",
    "url_removed":        "i feel so happy today!",
    "emoji_removed":      "i feel so happy today!",
    "special_removed":    "i feel so happy today",
    "chat_expanded":      "i feel so happy today",
    "stopwords_removed":  "feel happy today",
    "lemmatised":         ["feel", "happy", "today"]
  },
  "negations_kept": [],
  "history": [...]
}
```

### `GET /stats`

Returns all model performance metrics and class distribution for the dashboard.

```json
{
  "model_stats": {
    "BiLSTM":  { "train": 96.3, "val": 94.5, "test": 95.2, "precision": 95.4, "recall": 95.2, "f1": 95.2 },
    "CNN":     { "train": 94.2, "val": 93.2, "test": 92.6, ... },
    "SVM":     { "train": 93.5, "val": 91.3, "test": 91.3, ... },
    "XGBoost": { "train": 92.9, "val": 90.5, "test": 90.6, ... }
  },
  "class_distribution": { ... },
  "dataset_size": 416809,
  "best_model": "BiLSTM (95.2% test accuracy)",
  "imbalance_ratio": "9.5:1 (Joy:Surprise)",
  "hard_pairs": ["Fear / Surprise", "Sadness / Love"]
}
```

### `GET /health`

```json
{ "status": "ok", "model": "model.pkl", "project": "MindTrace" }
```

---

## Web UI — Four Tabs

| Tab | Content |
|---|---|
| **🔮 Predict** | Text input with 8 example pills (one per emotion class + 2 negation examples). After prediction: animated emoji + confidence ring + intensity bar + word count, probability bars for all 6 emotions, 7-step NLP pipeline output with negation tokens highlighted in red |
| **📊 Dashboard** | 4 KPI cards · Grouped bar chart (all 4 metrics) · Performance table with mini bars · Radar chart · Class distribution (Table 3.3) with animated inline bars + horizontal chart · BiLSTM training curve · Live session doughnut chart (updates as you predict) |
| **🕓 History** | Session prediction log with emotion summary chips (×count per label) and clear-all button. Badge counter on the tab itself |
| **📄 About** | Research question (Section 1.3) · Research gap (Section 1.2) · Dataset justification (Section 3.1) · NLP pipeline steps (Section 3.4) · All 4 model descriptions (Section 4) · 5 key EDA findings (Section 3.2) · 4 contribution points (Section 6.1) |

**Interactive features:** Confetti animation on high-confidence Joy (≥72%) · Gold star burst on Surprise (≥62%) · Keyboard shortcut `Ctrl+Enter` to predict · Hovering on emotion floaters pauses their animation

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Flask/Gunicorn listen port |
| `MODEL_PATH` | `model.pkl` | Path to serialised sklearn pipeline |

---

## Dependencies

```
flask==3.0.3          # Web framework
scikit-learn==1.4.2   # TF-IDF vectoriser + SVM pipeline
nltk==3.8.1           # Stopwords + lemmatiser
emoji==2.12.1         # Emoji removal from text
joblib==1.4.2         # Model serialisation / deserialisation
numpy==1.26.4         # Array operations
pandas==2.2.2         # DataFrame handling (training only)
gunicorn==22.0.0      # Production WSGI server (Docker)
openpyxl==3.1.2       # Read .xlsx dataset (training only)
xgboost==2.0.3        # XGBoost baseline (training only)
```

---

## Reproducing the Results

The `train_pipeline.py` script trains the **SVM** model deployed in this app. To reproduce the BiLSTM (95.2%) or CNN (92.6%) results, use the full `Emotion_prediction_source_code.ipynb` notebook in a TensorFlow 3.11 environment (Google Colab with NVIDIA T4 GPU recommended).

The sklearn pipeline produced by `train_pipeline.py` should achieve approximately **91.3% test accuracy** on the balanced dataset, matching the SVM row in Table 4.1.

---

## Citation

If you use this work, please cite:

> Aye Khin Khin Hpone (Yolanda Lim), *MindTrace: Text Mining and NLP-Driven Emotion Prediction Using Machine Learning and Deep Learning Approaches*, Asian Institute of Technology, Computer Science, March 2026.

---

*Asian Institute of Technology · Computer Science · March 2026*
