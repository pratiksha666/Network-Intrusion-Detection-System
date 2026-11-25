# Network Intrusion Detection System (NIDS) using Machine Learning

A beginner-friendly web application that classifies network traffic as **Normal (0)** or **Attack (1)** using the **UNSW-NB15** dataset.

- Frontend: Streamlit
- Backend: Python
- Machine Learning: Scikit-learn (Random Forest, KNN, SVM, Logistic Regression)

## Folder Structure

```
Network-Intrusion-Detection/
├── app.py                  # Streamlit frontend (all pages)
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                # put the UNSW-NB15 CSV here (optional, you can also upload in the app)
│   └── processed/
├── models/                 # trained models (.joblib) are saved here
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # load + validate CSV, missing values
│   ├── preprocessing.py    # cleaning, split, ColumnTransformer
│   ├── train_model.py      # build pipelines, train, save with Joblib
│   ├── predict.py          # load saved model, predict Normal/Attack
│   └── evaluation.py       # metrics, confusion matrix, comparison chart
└── assets/
```

## Setup and Run

```bash
cd Network-Intrusion-Detection
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Download the **UNSW-NB15 training and testing set CSVs** (`UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`) from the official UNSW Canberra page or Kaggle.
Do not use the four raw `UNSW-NB15_1..4.csv` files: they have no header row.

The CSV must contain `label` (0 = Normal, 1 = Attack) and the categorical columns `proto`, `service`, `state`.

## How to Use the App

1. **Upload Dataset** - upload the CSV, check rows, columns, missing values.
2. **Data Preprocessing** - cleans data and makes a stratified train/test split.
3. **Model Training** - train one model or all four. Models are saved in `models/`.
4. **Model Evaluation** - accuracy, precision, recall, F1, confusion matrix, comparison chart.
5. **Network Traffic Prediction** - predict on a test-set sample or a new CSV.
6. **Download Results** - download predictions as CSV.


