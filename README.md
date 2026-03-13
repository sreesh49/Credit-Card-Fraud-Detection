💳 Credit Card Fraud Detection using Machine Learning


📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Fraud detection is a highly imbalanced classification problem, where fraudulent transactions represent a very small percentage of the total transactions.

The goal of this project is to build and evaluate multiple machine learning models to accurately detect fraudulent transactions while minimizing false positives.

📊 Dataset

The dataset used in this project is the Credit Card Fraud Detection Dataset.

Dataset Characteristics

Total Transactions: 284,807

Fraudulent Transactions: 492

Features: 30

Target Variable:

0 → Non-Fraud

1 → Fraud

Most features are PCA-transformed except:

Time

Amount

Class

Because fraud cases are rare, the dataset is highly imbalanced.

⚙️ Technologies Used
Programming Language

Python

Libraries

pandas

numpy

matplotlib

seaborn

scikit-learn

📂 Project Workflow
1️⃣ Data Loading

Load the dataset and inspect its structure.

df = pd.read_csv("creditcard.csv")
df.head()


2️⃣ Exploratory Data Analysis (EDA)

Key analysis performed:

Transaction distribution

Fraud vs Non-fraud comparison

Feature distribution

Transaction amount analysis

Visualization of class imbalance

Fraud transactions represent less than 0.2% of the dataset, making this a highly imbalanced classification problem.

3️⃣ Data Preprocessing

Steps performed:

Feature scaling using StandardScaler

Train-test split

Handling class imbalance

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))


4️⃣ Handling Imbalanced Data (Undersampling)

Since fraudulent transactions are extremely rare, Random Undersampling was used to balance the dataset by reducing the number of non-fraudulent transactions.

This technique helps the model learn patterns from both classes more effectively.

Example approach:

fraud = df[df['Class'] == 1]
non_fraud = df[df['Class'] == 0].sample(len(fraud), random_state=42)

balanced_df = pd.concat([fraud, non_fraud])

After undersampling, the dataset becomes balanced between fraud and non-fraud transactions.

5️⃣ Machine Learning Models

The following machine learning models were trained and compared:

Logistic Regression

Support Vector Machine (SVM)

Naive Bayes

Random Forest

Dummy Classifier (baseline model)

6️⃣ Model Evaluation

Models were evaluated using the following metrics:

Confusion Matrix

Precision

Recall

F1 Score

ROC-AUC Curve

Precision-Recall Curve

Example metrics used:

precision_score()
recall_score()
f1_score()
roc_auc_score()
📈 ROC Curve Comparison

ROC curves were plotted to compare the performance of multiple models.

plot_roc_curves(X_test, y_test, models, model_names)

A higher ROC-AUC score indicates better classification performance.

🎯 Key Insights

Fraud detection requires high recall to capture as many fraudulent transactions as possible.

Undersampling helped address the class imbalance problem.

Random Forest and Logistic Regression performed well in detecting fraud patterns.

📊 Evaluation Metrics Importance
Metric	Importance in Fraud Detection
Precision	Reduces false fraud alerts
Recall	Detects maximum fraud cases
F1 Score	Balances precision and recall
ROC-AUC	Measures overall model performance
📁 Project Structure
credit-card-fraud-detection
│
├── creditcard.csv
├── fraud_detection_credit_card.ipynb
├── README.md
└── images
🚀 How to Run the Project


1️⃣ Clone the repository

git clone https://github.com/yourusername/credit-card-fraud-detection.git


2️⃣ Install dependencies

pip install pandas numpy scikit-learn matplotlib seaborn


3️⃣ Run the notebook


jupyter notebook

Open the notebook:

fraud_detection_credit_card.ipynb


📌 Future Improvements

Implement advanced models such as:

XGBoost

LightGBM

CatBoost

Perform hyperparameter tuning

Analyze feature importance

Deploy the fraud detection model as an API

👤 Author

Sreesh Sreekumar

Aspiring Data Analyst / Machine Learning Enthusiast
