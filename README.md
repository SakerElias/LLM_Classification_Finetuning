# 🧠 LLM_Classification_Finetuning

This repository contains our work for the **Machine Learning Project Course**, where we try to solve a **text classification** challenge from this [Kaggle competition](https://www.kaggle.com/competitions/llm-classification-finetuning).  

The goal is to explore how pretrained transformer-based language models can be efficiently adapted for **human preference prediction** in LLM-generated responses, using modern fine-tuning and lightweight modeling strategies.

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/SakerElias/LLM_Classification_Finetuning.git
cd LLM_Classification_Finetuning
```

Create a virtual environment (recommended: **Python 3.10**):
```bash
python -m venv .venv
```

Activate the environment:

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧩 Project Structure

```
LLM_Classification_Finetuning/
│
├── data/                     # train.csv, test.csv, sample_submission.csv
├── notebooks/                # EDA and modeling notebooks
├── requirements.txt
├── README.md
└── report.pdf
```

---

## 📊 Modeling Summary

The baseline model uses simple lexical and structural features (e.g., length, paragraph count, list usage, quotes) identified through exploratory data analysis (EDA).  
We train a multinomial **Logistic Regression** model using **Stratified K-Fold Cross-Validation** to ensure balanced evaluation across classes.

**Sentence-transformers/all-MiniLM-L6-v2** is used to generate embeddings model. Prompt-response pairs are created ,and responses are concatenated into a single feature vector. A **Logistic Regression** classifier trained on these embeddings. Embedding model have a similar performance to the lexical baseline.