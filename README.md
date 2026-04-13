# MindTrace — NLP-Driven Emotion Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Flask-3.0.3-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/scikit--learn-1.4.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Test_Accuracy-93.9%25_(BiLSTM)-22c55e?style=flat-square" alt="Accuracy">
</p>

<p align="center">
  <b>Author:</b> Aye Khin Khin Hpone (Yolanda Lim) &nbsp;·&nbsp; ST125970<br>
  <b>Programme:</b> Computer Science, Asian Institute of Technology<br>
  <i>MindTrace: Text Mining and NLP-Driven Emotion Prediction Using Machine Learning and Deep Learning Approaches</i>
</p>

<p align="center">
  <img src="figures/demo.gif" alt="MindTrace Demo" width="608">
</p>

<p align="center">
  <a href="http://192.41.170.112:5970/">Live Demo</a> &nbsp;·&nbsp;
  <a href="https://youtu.be/BuqDFtVFqBw">YouTube Demo</a> &nbsp;·&nbsp;
  <a href="https://hub.docker.com/r/yolandalim/125970-mindtrace">Docker Hub</a> &nbsp;·&nbsp;
  <a href="https://github.com/limhpone/st125970-MindTrace-yolanda-ML-Final-Project">Source Code</a>
</p>

---

## Table of Contents

1. [Research Question](#research-question)
2. [Model Performance](#model-performance)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [NLP Preprocessing Pipeline](#nlp-preprocessing-pipeline)
6. [Quick Start](#quick-start)
7. [API Reference](#api-reference)
8. [Web UI](#web-ui)
9. [Environment Variables](#environment-variables)
10. [Dependencies](#dependencies)
11. [Reproducing Results](#reproducing-results)
12. [Citation & Acknowledgements](#citation--acknowledgements)

---

## Research Question

> How do traditional ML models (SVM, XGBoost) compare with deep learning models (CNN, BiLSTM) in multi-class emotion classification from social media text, when evaluated under a unified preprocessing pipeline, standardised dataset partitioning, and consistent evaluation metrics?

---

## Model Performance

*All models trained on the same preprocessing pipeline and dataset split.*

| Model | Train Acc. | Val Acc. | Test Acc. | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BiLSTM** *(best)* | 94.7% | 94.2% | **93.9%** | 94.1% | 93.9% | 93.9% |
| CNN | 93.2% | 92.8% | 92.5% | 92.7% | 92.5% | 92.5% |
| SVM | 94.4% | 92.1% | 91.8% | 92.0% | 91.8% | 91.8% |
| XGBoost | 93.3% | 91.1% | 90.8% | 91.0% | 90.8% | 90.7% |

> **Deployed model:** SVM + TF-IDF sklearn Pipeline — CPU inference, no GPU required, instant startup.

---

## Project Structure

```
mindtrace/
├── app.py                        # Flask API — loads model.pkl, serves UI
├── train_pipeline.py             # Trains and saves model.pkl from the dataset
├── model.pkl                     # Serialised sklearn Pipeline (SVM + TF-IDF)
├── requirements.txt              # Pinned Python dependencies
├── Dockerfile                    # Container definition (python:3.11-slim + gunicorn)
│
├── templates/
│   └── index.html                # MindTrace web UI (4 tabs — self-contained)
│
├── Emotion_prediction_source_code.ipynb   # Full training notebook (CNN / BiLSTM)
├── test_predict.ipynb            # Inference tests — 6 classes + batch prediction
├── ablation_study.ipynb          # Ablation study notebook
│
└── data/
    └── text.xlsx                 # Raw dataset — download separately from Kaggle
```

---

## Dataset

**Source:** [Kaggle — Emotions Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions)  
416,809 labelled Twitter samples · 6 emotion classes · Imbalance ratio 9.5:1

*Class distribution (after Eq. 1 in the paper).*

| Label | Code | Count | % | Note |
|---|:---:|---:|:---:|---|
| Joy | 1 | 141,067 | 33.84% | Largest class — risk of bias if unweighted |
| Sadness | 0 | 121,187 | 29.07% | Semantically close to Fear and Love |
| Anger | 3 | 57,317 | 13.75% | Distinct vocabulary aids separation |
| Fear | 4 | 47,712 | 11.45% | Often confused with Surprise |
| Love | 2 | 34,554 | 8.29% | Minority class |
| Surprise | 5 | 14,972 | 3.59% | Smallest — highest misclassification risk |

**High-risk confusion pairs:** Fear ↔ Surprise · Joy ↔ Love

---

## NLP Preprocessing Pipeline

Applied identically in `train_pipeline.py` during training and in `app.py` at inference time. Any divergence between the two causes training/serving skew.

*8-step pipeline (Section 3.3 — System Architecture in the paper).*

| Step | Operation | Detail |
|:---:|---|---|
| 1 | Lowercase | `str.lower()` |
| 2 | Whitespace stripping | Collapse multiple spaces, strip leading/trailing |
| 3 | URL removal | `re.sub(r'http\S+\|www\S+', '', t)` |
| 4 | Emoji removal | `emoji.replace_emoji(t, replace='')` |
| 5 | Special character removal | Keep only `[a-z\s]` |
| 6 | Chat word expansion | `u→you`, `lol→laugh out loud`, 31 rules |
| 7 | Stopword removal | NLTK English stopwords — **negation words preserved** |
| 8 | Lemmatisation | `nltk.WordNetLemmatizer` |

**Negation words preserved** (Section 3.3):  
`not` · `never` · `no` · `nor` · `neither` · `nothing` · `nobody` · `nowhere` · `without` · `very` · `extremely` · `barely` · `hardly`

These carry emotional polarity — removing them would lose discriminative signal.

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

# 4. Train and save model.pkl  (~5–15 min depending on hardware)
python train_pipeline.py --data data/text.xlsx

# 5. Start the server
python app.py
#    → Open http://localhost:5000
```

### Option 2 — Docker (build locally)

```bash
# model.pkl must exist first — run step 4 above
docker build -t mindtrace .
docker run -p 5000:5000 mindtrace
#    → Open http://localhost:5000
```

### Option 3 — Docker Hub (no build required)

```bash
docker pull yolandalim/125970-mindtrace:latest
docker run -p 5000:5000 yolandalim/125970-mindtrace:latest
#    → Open http://localhost:5000
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
    { "label": "Joy",      "probability": 87.43, "emoji": "😊", "color": "#0fffa0" },
    { "label": "Love",     "probability":  6.12, "emoji": "❤️", "color": "#06b6d4" },
    { "label": "Sadness",  "probability":  3.21, "emoji": "😢", "color": "#14b8a6" },
    { "label": "Surprise", "probability":  1.45, "emoji": "😲", "color": "#fbbf24" },
    { "label": "Anger",    "probability":  1.02, "emoji": "😠", "color": "#f87171" },
    { "label": "Fear",     "probability":  0.77, "emoji": "😨", "color": "#a78bfa" }
  ],
  "cleaned_text": "feel happy today",
  "tokens":       ["feel", "happy", "today"],
  "nlp_steps": {
    "lowercased":          "i feel so happy today!",
    "whitespace_stripped": "i feel so happy today!",
    "url_removed":         "i feel so happy today!",
    "emoji_removed":       "i feel so happy today!",
    "special_removed":     "i feel so happy today",
    "chat_expanded":       "i feel so happy today",
    "stopwords_removed":   "feel happy today",
    "lemmatised":          ["feel", "happy", "today"]
  },
  "negations_kept": [],
  "history": [...]
}
```

---

### `GET /stats`

Returns all model performance metrics and class distribution for the dashboard.

```json
{
  "model_stats": {
    "BiLSTM":  { "train": 94.7, "val": 94.2, "test": 93.9, "precision": 94.1, "recall": 93.9, "f1": 93.9 },
    "CNN":     { "train": 93.2, "val": 92.8, "test": 92.5, "..." : "..." },
    "SVM":     { "train": 94.4, "val": 92.1, "test": 91.8, "..." : "..." },
    "XGBoost": { "train": 93.3, "val": 91.1, "test": 90.8, "..." : "..." }
  },
  "class_distribution": { "...": "..." },
  "dataset_size":    416809,
  "best_model":      "BiLSTM (93.9% test accuracy)",
  "imbalance_ratio": "9.5:1 (Joy:Surprise)",
  "hard_pairs":      ["Fear / Surprise", "Joy / Love"]
}
```

---

### `GET /health`

```json
{ "status": "ok", "model": "model.pkl", "project": "MindTrace" }
```

---

## Web UI

The app ships a single-page, self-contained UI with four tabs.

| Tab | Content |
|---|---|
| **Predict** | Text input with 8 example pills (one per emotion + 2 negation examples). After prediction: animated emoji, confidence ring, intensity bar, word count, probability bars for all 6 emotions, and the 8-step NLP trace with negation tokens highlighted. |
| **Dashboard** | 4 KPI cards · Grouped bar chart (all 4 metrics) · Performance table with mini bars · Radar chart · Class distribution with animated inline bars + horizontal chart · BiLSTM training curve · Live session doughnut (updates as you predict). |
| **History** | Session prediction log with emotion summary chips (×count per label) and clear-all button. Badge counter on the tab itself. |
| **About** | Research question · Research gap · Dataset justification · NLP pipeline · All 4 model descriptions · 5 key EDA findings · 3 contribution points. |

**Interactive features:**
- Confetti animation on high-confidence Joy predictions (≥ 72%)
- Gold star burst on Surprise predictions (≥ 62%)
- Keyboard shortcut `Ctrl+Enter` to predict
- Hovering on emotion floaters pauses their animation

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Flask / Gunicorn listen port |
| `MODEL_PATH` | `model.pkl` | Path to the serialised sklearn pipeline |

---

## Dependencies

| Package | Version | Used for |
|---|---|---|
| `flask` | 3.0.3 | Web framework |
| `scikit-learn` | 1.4.2 | TF-IDF vectoriser + SVM pipeline |
| `nltk` | 3.8.1 | Stopwords + lemmatiser |
| `emoji` | 2.12.1 | Emoji removal from text |
| `joblib` | 1.4.2 | Model serialisation / deserialisation |
| `numpy` | 1.26.4 | Array operations |
| `gunicorn` | 22.0.0 | Production WSGI server (Docker) |
| `pandas` | 2.2.2 | DataFrame handling *(training only)* |
| `openpyxl` | 3.1.2 | Read `.xlsx` dataset *(training only)* |
| `xgboost` | 2.0.3 | XGBoost baseline *(training only)* |

> `pandas`, `openpyxl`, and `xgboost` are commented out in `requirements.txt` by default — uncomment them when running `train_pipeline.py`.

---

## Reproducing Results

`train_pipeline.py` trains the **SVM** model deployed in this app and should achieve approximately **91.8% test accuracy**, matching the SVM row in Table 9.

To reproduce the **BiLSTM (93.9%)** or **CNN (92.5%)** results, use the full `Emotion_prediction_source_code.ipynb` notebook in a TensorFlow environment (Google Colab with NVIDIA T4 GPU recommended).

---

## Citation & Acknowledgements

**Citation**

> Aye Khin Khin Hpone (Yolanda Lim), *MindTrace: Text Mining and NLP-Driven Emotion Prediction Using Machine Learning and Deep Learning Approaches*, Asian Institute of Technology, Computer Science, March 2026.

**Acknowledgements**

Sincere thanks to **Dr. Sein Minn** for his supervision and guidance, and to **TA Rakshya Lama Moktan** for her valuable support and feedback.

---

*Asian Institute of Technology · Computer Science · March 2026*
