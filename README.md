# Network Intrusion Detection System (NIDS) using Machine Learning

A beginner-friendly web application that classifies network traffic as **Normal (0)** or **Attack (1)** using the **UNSW-NB15** dataset.

- Frontend: Streamlit
- Backend: Python
- Machine Learning: Scikit-learn (Random Forest, KNN, SVM, Logistic Regression)

## Folder Structure

```
Network-Intrusion-Detection/
├── app.py                  
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                
│   └── processed/
├── models/                
├── src/
│   ├── __init__.py
│   ├── data_loader.py     
│   ├── preprocessing.py    
│   ├── train_model.py      
│   ├── predict.py          
│   └── evaluation.py      
└── assets/
```

## Setup and Run

```bash
cd Network-Intrusion-Detection
python -m venv venv
venv\Scripts\activate          
# source venv/bin/activate     
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

## Future Scope
1.Implement real-time network monitoring./
2.Improve accuracy using advanced ML algorithms./
3.Add automatic security alerts./
4.Deploy the system on cloud platforms./
5.Detect more types of cyberattacks./   

## Project Objectives
1.To detect suspicious network traffic using Machine Learning.
2.To classify normal and malicious network activities.
3.To use Random Forest for intrusion detection.

