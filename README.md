# Network Intrusion Detection System (NIDS) Using Machine Learning

A beginner-friendly web application that classifies network traffic as **Normal (0)** or **Attack (1)** using the **UNSW-NB15 dataset**.

## Technologies Used

- **Frontend:** Streamlit
- **Backend:** Python
- **Machine Learning:** Scikit-learn
- **Algorithms:** Random Forest, KNN, SVM, Logistic Regression
- **Dataset:** UNSW-NB15

## Project Folder Structure

```text
Network-Intrusion-Detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── evaluation.py
│
└── assets/
```

## Setup and Run

Follow these steps to set up and run the project locally.

### 1. Clone or Navigate to the Project Directory

```bash
cd Network-Intrusion-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**For Windows:**

```bash
venv\Scripts\activate
```

**For Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

## Dataset

This project uses the **UNSW-NB15 dataset** for network intrusion detection.

Download the training and testing CSV files:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

You can download the dataset from the official UNSW Canberra website or Kaggle.

**Important:**

- Use the training and testing CSV files mentioned above.
- Do not use the four raw files (`UNSW-NB15_1.csv` to `UNSW-NB15_4.csv`) directly, as they do not contain header rows.
- The dataset must contain a `label` column:
  - `0` = Normal Traffic
  - `1` = Attack Traffic
- The categorical columns should include `proto`, `service`, and `state`.

## How to Use the Application

### 1. Upload Dataset
- Upload the dataset in CSV format.
- View the number of rows and columns.
- Check missing values and dataset information.

### 2. Data Preprocessing
- Clean the dataset.
- Handle missing values.
- Encode categorical features.
- Perform a stratified train-test split.

### 3. Model Training
- Select and train individual machine learning models.
- Alternatively, train all four models.
- Save trained models in the `models/` directory.

### 4. Model Evaluation
Evaluate model performance using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Model Comparison Chart

### 5. Network Traffic Prediction
- Predict network traffic using a test dataset sample.
- Upload a new CSV file for prediction.
- Classify network traffic as Normal or Attack.

### 6. Download Results
- Download prediction results in CSV format.
- Use the results for further analysis.

## Project Objectives

1. To detect suspicious network traffic using Machine Learning.
2. To classify normal and malicious network activities.
3. To implement the Random Forest algorithm for intrusion detection.
4. To evaluate different machine learning algorithms for network security.
5. To develop a simple and interactive web application for intrusion detection.

## Future Scope

1. Implement real-time network traffic monitoring.
2. Improve detection performance using advanced machine learning algorithms.
3. Add automatic security alerts for suspicious network activities.
4. Deploy the application on cloud platforms.
5. Extend the system to identify different types of cyberattacks.

---

**Note:** This project is developed for educational purposes to demonstrate the application of Machine Learning in network intrusion detection.
