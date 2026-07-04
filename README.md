# 💳 Credit Card Fraud Detection using Machine Learning

A Machine Learning project designed to identify fraudulent credit card transactions using transaction data and classification algorithms. The goal of this project is to build a predictive system capable of distinguishing normal transactions from potentially fraudulent activities with high accuracy.

---

## 📌 Project Overview

Credit card fraud has become one of the most critical challenges in the financial sector. As online transactions continue to grow worldwide, detecting fraudulent activities quickly and accurately is essential for protecting users and financial institutions.

This project applies Machine Learning techniques to analyze transaction patterns and classify whether a transaction is legitimate or fraudulent.

The project follows a complete machine learning pipeline:

- Data Collection
- Data Preprocessing
- Data Analysis
- Handling Imbalanced Data
- Model Training
- Model Evaluation
- Prediction System Development

---

## 🎯 Project Objectives

The major goals of this project are:

✅ Analyze credit card transaction data

✅ Handle highly imbalanced datasets

✅ Train a machine learning classification model

✅ Evaluate model performance

✅ Predict fraudulent transactions

✅ Build an end-to-end fraud detection workflow

---

## 📊 Dataset Information

The dataset used for this project contains anonymized credit card transactions made by European cardholders.

### Dataset Features

The dataset contains:

- **Time** → Time elapsed between transactions
- **V1–V28** → Features transformed using PCA (Principal Component Analysis)
- **Amount** → Transaction amount
- **Class** → Target variable

Target variable:

```text
0 → Legit Transaction
1 → Fraudulent Transaction
```

---

## 📈 Dataset Characteristics

- Total Transactions: **284,807**
- Fraud Cases: **492**
- Legit Transactions: **284,315**
- Dataset Type: Highly Imbalanced Dataset

Class distribution:

```text
Legit Transactions      → 284315
Fraud Transactions      → 492
```

Because fraud transactions are extremely rare, handling class imbalance becomes one of the biggest challenges in this project.

---

## 🛠 Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

### Development Tools

- VS Code
- Git
- GitHub

---

## 📂 Project Structure

```text
Credit_Card_Fraud_Prediction/
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── dataset_link.txt
```

---

## ⚙️ Project Workflow

### Step 1: Importing Libraries

Required libraries were imported:

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

### Step 2: Loading Dataset

Dataset loaded using Pandas:

```python
credit_card_data = pd.read_csv("creditcard.csv")
```

---

### Step 3: Exploratory Data Analysis

Data exploration included:

- Dataset shape
- Missing values
- Statistical summary
- Fraud transaction distribution
- Correlation analysis

Example:

```python
print(credit_card_data.head())
print(credit_card_data.describe())
print(credit_card_data['Class'].value_counts())
```

---

### Step 4: Handling Imbalanced Data

Since fraud cases are much smaller than normal transactions, balanced sampling was used.

Example:

```python
legit = credit_card_data[
    credit_card_data.Class == 0
]

fraud = credit_card_data[
    credit_card_data.Class == 1
]

legit_sample = legit.sample(
    n=492
)

new_dataset = pd.concat(
    [legit_sample, fraud],
    axis=0
)
```

---

### Step 5: Feature and Target Separation

```python
X = new_dataset.drop(
    columns='Class',
    axis=1
)

Y = new_dataset['Class']
```

---

### Step 6: Train-Test Split

Dataset divided into training and testing datasets:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)
```

---

### Step 7: Model Training

Logistic Regression was used:

```python
model = LogisticRegression()

model.fit(
    X_train,
    Y_train
)
```

---

## 🧠 Why Logistic Regression?

Logistic Regression is commonly used for binary classification because:

- Fast and efficient
- Easy to interpret
- Performs well on classification tasks
- Generates probability estimates
- Works effectively with balanced datasets

---

## 📈 Model Evaluation

The model performance was evaluated using:

### Accuracy Score

Training Accuracy:

```python
training_data_accuracy = accuracy_score(
    Y_train,
    X_train_prediction
)
```

Testing Accuracy:

```python
testing_data_accuracy = accuracy_score(
    Y_test,
    X_test_prediction
)
```

---

## 💳 Fraud Prediction System

A predictive system was implemented where users can input transaction values and determine whether a transaction is fraudulent.

Example:

```python
input_data = np.asarray([
    transaction_values
])

input_data_reshaped = input_data.reshape(
    1,-1
)

prediction = model.predict(
    input_data_reshaped
)

if prediction[0]==0:
    print("Legitimate Transaction")

else:
    print("Fraudulent Transaction")
```

Example Output:

```text
Legitimate Transaction
```

or

```text
Fraudulent Transaction
```

---

## 🚀 Installation and Setup

Clone repository:

```bash
git clone https://github.com/Hellboy28D/Credit_Card_Fraud_Prediction.git
```

Move into project directory:

```bash
cd Credit_Card_Fraud_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python src/predict.py
```

---

## 📊 Dataset Source

Due to GitHub file size limitations, the dataset is not included in this repository.

Dataset can be downloaded from:

Dataset Source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 📈 Future Improvements

Future enhancements may include:

- Random Forest Classifier
- XGBoost
- Deep Learning approaches
- Hyperparameter tuning
- Real-time fraud detection API
- Streamlit web app
- Model deployment
- Cloud integration

---

## 💡 Key Learnings

This project improved my understanding of:

- Machine Learning workflow
- Classification problems
- Imbalanced datasets
- Logistic Regression
- Fraud detection systems
- Data preprocessing
- Model evaluation
- GitHub project organization
- Debugging and problem solving

---

## 👨‍💻 Author

**Divakr Dayas**

GitHub:

https://github.com/Hellboy28D

---

## ⭐ Support

If you found this project useful, consider giving it a star ⭐

Suggestions and feedback are always welcome.

Happy Coding 🚀
