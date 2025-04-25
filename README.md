
# Medical Appointment No-Show Prediction

This project aims to predict whether a patient will show up for their medical appointment using various machine learning techniques. Models compared include Logistic Regression, SVM, Random Forest, Neural Networks, and XGBoost.

---

## ⚙️ Setup

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run notebooks in sequence**
   - `01_eda.ipynb`: Data Exploration
   - `02_data_preprocessing.ipynb`: Data Cleaning and Feature Engineering
   - `03_model_development.ipynb`: Model Training and Comparison
   - `04_nn_experiment.ipynb`: (Included into 03)

---

## 🚀 Key Features

- Robust data preprocessing pipeline
- Feature engineering and handling of missing values
- Imbalanced class treatment with SMOTE and `class_weight`
- Cost-sensitive evaluation metric design
- Model benchmarking using business-specific KPIs
- Neural network experiments with hyperparameter tuning

---

## 🤖 Models Evaluated

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Neural Networks (PyTorch)
- XGBoost

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- ROC-AUC
- **Cost Savings** *(based on intervention vs. no-show cost)*

---

## 📦 Requirements

- Python 3.12.7
- [PyTorch](https://pytorch.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- Pandas, NumPy
- Matplotlib, Seaborn
- [Weights & Biases (wandb)](https://wandb.ai/)


