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

Download the trainset (train.csv) from the kaggle competition and put it inside data folder as train.csv
---

## 🧩 Project Structure

```
LLM_Classification_Finetuning/
│
├── artifacts/                # contains saved embeddings for models
├── data/                     # train.csv, test.csv, sample_submission.csv
├── notebooks/                # EDA and modeling notebooks
├── outputs/                  # contains submissions for each step (csv files)
├── requirements.txt
├── README.md
└── report.pdf
```

---

## 📊 Notebooks & Modeling Summary

**EDA**
In this notebook we perform a quick Exploratory Data Analysis to try to understand the nature of our dataset and which features could be useful for modelling, especially in step 1

**step1.ipynb**
The baseline model uses simple lexical and structural features (e.g., length, paragraph count, list usage, quotes) identified through exploratory data analysis (EDA).  
We train a multinomial **Logistic Regression** model using **Stratified K-Fold Cross-Validation** to ensure balanced evaluation across classes.

**step2.ipynb**
**all-MiniLM-L6-v2** is used to generate embeddings model. Prompt-response pairs are created ,and responses are concatenated into a single feature vector. A **Logistic Regression** classifier trained on these embeddings. Embedding model have a similar performance to the lexical baseline.

**step3.ipynb**
This notebook allows to choose between 4 modeling options : 
    Use a calibrated Logistic Regression model with added lexical features compared to step 1
    Use an upgraded version of **all-MiniLM-L6-v2** with a calibrated classifier on top
    Use LoRA to fine-tune a larger embedding model : **DeBERTa** and use it with a classifier
    Use ensembling techniques to mix between the first and the second model.

**step3 (kaggle version)** is an adapted version of step 3 to Kaggle's environment and restrictions for submission (no internet rule typically)

**step4.ipynb**
In this notebook we perform a deeper Error and Bias Analysis of three of our models from step 3 (Lexical (Isotonic), Embeddings (Isotonic), and Ensemble (Weighted + Temperature Scaling))