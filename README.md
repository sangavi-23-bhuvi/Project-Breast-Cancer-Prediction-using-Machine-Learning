# Breast Cancer Prediction using Machine Learning

A machine learning project that classifies breast tumors as **malignant** or **benign** using the Wisconsin Breast Cancer dataset (569 samples, 30 numeric features). Four classifiers are compared and the best one is saved for reuse.

## Highlights
- Exploratory data analysis: class balance and feature correlation heatmap
- Leak-free preprocessing using scikit-learn `Pipeline` (scaling fit on training data only)
- Models compared: Logistic Regression, KNN, SVM (RBF), Random Forest
- 5-fold cross-validation plus held-out test evaluation (accuracy, precision, recall, F1)
- Confusion matrix, saved model (`joblib`) and a sample prediction

## Results (80/20 stratified split)

| Model | CV Accuracy | Test Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.9802 | 0.9825 | 0.9861 | 0.9861 | 0.9861 |
| SVM | 0.9714 | 0.9825 | 0.9861 | 0.9861 | 0.9861 |
| KNN | 0.9714 | 0.9737 | 0.9600 | 1.0000 | 0.9796 |
| Random Forest | 0.9604 | 0.9561 | 0.9589 | 0.9722 | 0.9655 |

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib

## How to Run
```bash
git clone https://github.com/<your-username>/breast-cancer-prediction.git
cd breast-cancer-prediction
pip install -r requirements.txt
python main.py
```
Plots, the model comparison table and the trained model are saved in the `outputs/` folder.

To run it as a notebook instead, open `Breast_Cancer_Prediction.ipynb` in Jupyter Notebook, JupyterLab, VS Code or Google Colab and choose **Run All**.

## Project Structure
```
breast-cancer-prediction/
├── Breast_Cancer_Prediction.ipynb  # notebook version with outputs
├── main.py            # full pipeline: EDA, training, evaluation, saving
├── requirements.txt
├── README.md
└── outputs/           # generated plots, CSV and saved model
```

## Disclaimer
This is an educational project and must not be used for real medical diagnosis.
