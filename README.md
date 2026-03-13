# Credit-Card-Fraud-Detection

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

imbalanced-learn (SMOTE)

📂 Project Workflow
1️⃣ Data Loading

Load dataset and inspect its structure.

df = pd.read_csv("creditcard.csv")
df.head()
2️⃣ Exploratory Data Analysis (EDA)

Key analysis performed:

Transaction distribution

Fraud vs Non-fraud analysis

Feature distribution

Time and Amount analysis

Class imbalance visualization

Example:

Fraud transactions represent less than 0.2% of the dataset.

3️⃣ Data Preprocessing

Steps performed:

Feature scaling using StandardScaler

Train-test split

Handling class imbalance

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
4️⃣ Handling Imbalanced Data

Since fraud cases are very rare, SMOTE (Synthetic Minority Oversampling Technique) is used.

sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
5️⃣ Machine Learning Models

The following models were trained and compared:

Logistic Regression

Support Vector Machine (SVM)

Naive Bayes

Random Forest

Dummy Classifier (baseline)

6️⃣ Model Evaluation

Models were evaluated using:

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

ROC curves were plotted to compare multiple models.

plot_roc_curves(X_test, y_test, models, model_names)

A higher ROC-AUC score indicates better classification performance.

🎯 Key Insights

Fraud detection requires high recall to capture fraudulent transactions.

Random Forest and Logistic Regression showed strong performance.

SMOTE significantly improved model performance on minority class.

📊 Evaluation Metrics Importance
Metric	Importance in Fraud Detection
Precision	Avoid false fraud alerts
Recall	Detect maximum fraud cases
F1 Score	Balance precision and recall
ROC-AUC	Overall model performance
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
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
3️⃣ Run the notebook
jupyter notebook

Open:

fraud_detection_credit_card.ipynb
📌 Future Improvements

Try advanced models like:

XGBoost

LightGBM

CatBoost

Hyperparameter tuning

Feature importance analysis

Deploy fraud detection model using API

👤 Author

Sreesh Sreekumar

Aspiring Data Analyst / Machine Learning Engineer
