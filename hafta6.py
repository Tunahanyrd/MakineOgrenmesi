#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
# Created on Thu Mar 27 13:01:04 2025
@author: tunahan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# Modeller
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# veri yolları
X_train = pd.read_csv("C:\\Users\\Tunahan\\.cache\\kagglehub\\datasets\\altruistdelhite04\\loan-prediction-problem-dataset\\versions\\1\\train_u6lujuX_CVtuZ9i.csv")
X_test = pd.read_csv("C:\\Users\\Tunahan\\.cache\\kagglehub\\datasets\\altruistdelhite04\\loan-prediction-problem-dataset\\versions\\1\\test_Y3wMUE5_7gLdaTN.csv")

# train için temizlik
X_train["Gender"] = X_train["Gender"].fillna(X_train["Gender"].mode()[0])
X_train["Married"] = X_train["Married"].fillna(X_train["Married"].mode()[0])
X_train["Dependents"] = X_train["Dependents"].fillna(X_train["Dependents"].mode()[0])
X_train["Self_Employed"] = X_train["Self_Employed"].fillna(X_train["Self_Employed"].mode()[0])
X_train["LoanAmount"] = X_train["LoanAmount"].fillna(X_train["LoanAmount"].median())
X_train["Loan_Amount_Term"] = X_train["Loan_Amount_Term"].fillna(X_train["Loan_Amount_Term"].mode()[0])
X_train["Credit_History"] = X_train["Credit_History"].fillna(X_train["Credit_History"].mode()[0])

# test için temizlik
X_test["Gender"] = X_test["Gender"].fillna(X_test["Gender"].mode()[0])
X_test["Dependents"] = X_test["Dependents"].fillna(X_test["Dependents"].mode()[0])
X_test["Self_Employed"] = X_test["Self_Employed"].fillna(X_test["Self_Employed"].mode()[0])
X_test["LoanAmount"] = X_test["LoanAmount"].fillna(X_test["LoanAmount"].median())
X_test["Loan_Amount_Term"] = X_test["Loan_Amount_Term"].fillna(X_test["Loan_Amount_Term"].mode()[0])

# dummies
kategorik_kolonlar = ["Gender", "Married", "Dependents", "Self_Employed", "Education", "Property_Area"]
X_train = pd.get_dummies(X_train, columns=kategorik_kolonlar, drop_first=True)
X_test = pd.get_dummies(X_test, columns=kategorik_kolonlar, drop_first=True)

# hedef sütunu
y = X_train["Loan_Status"].map({"N": 0, "Y": 1})
X_train = X_train.drop(columns=["Loan_ID", "Loan_Status"])
X_test = X_test.drop(columns=["Loan_ID", "Loan_Amount_Term"])

# test setinden de target çıkar çünkü biz sadece train ile modelleri karşılaştıracağız
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.25, random_state=42)

# modelleri tanımla
modeller = {
    "Logistic Regression": LogisticRegression(solver='liblinear', 
                                              C=1.0, max_iter=200),
    
    "Decision Tree": DecisionTreeClassifier(criterion="gini", 
                                            max_depth=6, 
                                            min_samples_split=10, 
                                            random_state=42),
    
    "Random Forest": RandomForestClassifier(n_estimators=150, 
                                            max_depth=8, 
                                            min_samples_split=5, 
                                            random_state=42),
    
    "Naive Bayes": GaussianNB(var_smoothing=1e-9),
    
    "SVM": SVC(kernel="rbf", C=1.5, 
               gamma="scale", 
               probability=True),
    
    "KNN": KNeighborsClassifier(n_neighbors=7, 
                                weights="uniform", 
                                metric="minkowski"),
    
    "XGB": XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        reg_lambda=15,
        reg_alpha=5,
        gamma=2.5,
        colsample_bytree=0.75,
        n_estimators=300,
        use_label_encoder=False,
        random_state=42
    )
}


# sonuçlar tutulacak
sonuclar = []

# grafik için confusion matrixler
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.ravel()

for i, (isim, model) in enumerate(modeller.items()):
    clf = make_pipeline(StandardScaler(), model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    rapor = classification_report(y_val, y_pred, output_dict=True)

    sonuclar.append({
        "Model": isim,
        "Accuracy": acc,
        "Precision": rapor["1"]["precision"],
        "Recall": rapor["1"]["recall"],
        "F1-Score": rapor["1"]["f1-score"]
    })

    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma", ax=axes[i])
    axes[i].set_title(f"{isim}")
    axes[i].set_xlabel("Tahmin")
    axes[i].set_ylabel("Gerçek")

plt.tight_layout()
plt.show()

# metrikleri tabloya dök
df_sonuclar = pd.DataFrame(sonuclar)
print(df_sonuclar)

# accuracy karşılaştırma
plt.figure(figsize=(10,6))
sns.barplot(x="Model", y="Accuracy", data=df_sonuclar)
plt.ylim(0, 1)
plt.title("Modellere Göre Doğruluk (Accuracy)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
