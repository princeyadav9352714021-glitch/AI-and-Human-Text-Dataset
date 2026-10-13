# AI vs Human Text Detector — Frontend

This is a Streamlit frontend for the trained TF-IDF + Linear SVM model from `AI_and_Human_Text_Dataset.ipynb`.
---
![App Demo](video.gif)

---
## Folder structure

```text
ai_human_frontend/
├── app.py
├── requirements.txt
└── ai_human_svm_model.pkl   # copy this from your trained notebook
```

## 1. Export model from your notebook

Run this after training `svm_model`:

```python
import joblib
joblib.dump(svm_model, "ai_human_svm_model.pkl")
```

Download the `.pkl` file from Colab and copy it beside `app.py`.

## 2. Open folder in VS Code

Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Start frontend:
```bash
streamlit run app.py
```

Streamlit opens the app in your browser.

## Notes
- The model expects text processed the same way as in the notebook; `app.py` uses the same cleaning, NLTK tokenization, stopword removal, and lemmatization.
- The model predicts `AI` or `Human`; it does not provide calibrated probability scores because LinearSVC does not implement `predict_proba`.
- Test accuracy from your notebook is not a guarantee for all real-world texts. Validate with varied, unseen examples.
