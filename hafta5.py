#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 21:10:54 2025

@author: tunahan
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats


# diamonds veri seti
df = sns.load_dataset("diamonds")
print("Veri seti bilgileri:")
print(df.info())
print("\nEksik değer sayıları:")
print(df.isnull().sum())

# Aykırı Değer Tespiti ve Temizleme (price sütunu için IQR tercih ettim)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR

# Aykırı 
df = df[(df['price'] >= alt_sinir) & (df['price'] <= ust_sinir)]


# korelasyon matrisi
num_df = df.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(10,8))
corr = num_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()

# encode
# 'cut', 'color' ve 'clarity' sütunları için one-hot encoding
df = pd.get_dummies(df, columns=["cut", "color", "clarity"], drop_first=True)


# Basit Lineer Regresyon: 'carat' --> 'price'
X_simple = df[['carat']]
y = df['price']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# eğitim
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)

y_pred_s = model_simple.predict(X_test_s)

# Sonuçların yazdırılması
print("\nBasit Lineer Regresyon Sonuçları:")
print("R^2 Skoru:", r2_score(y_test_s, y_pred_s))
print("MSE:", mean_squared_error(y_test_s, y_pred_s))

plt.figure(figsize=(8,6))
plt.scatter(X_test_s, y_test_s, alpha=0.5, label="Gerçek Değerler")
plt.plot(X_test_s, y_pred_s, color='red', label="Tahmin")
plt.xlabel("Carat")
plt.ylabel("Price")
plt.title("Basit Lineer Regresyon: Carat - Price")
plt.legend()
plt.show()

# Çoklu Lineer Regresyon: Tüm değişkenler kullanılarak 'price' tahmini
X_multi = df.drop(columns=["price"])
y_multi = df["price"]

# Eğitim ve test setlerine ayırma
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

# Test setinde tahmin
y_pred_m = model_multi.predict(X_test_m)

# Sonuçların yazdırılması
print("\nÇoklu Lineer Regresyon Sonuçları:")
print("R^2 Skoru:", r2_score(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))

# Normalizasyon ve d'agostino k^2 Testi ile dönüşüm karşılaştırması
data_original = df['price']
# dönüşümler
price_boxcox, lambda_boxcox = stats.boxcox(data_original)
price_log = np.log(data_original)
price_sqrt = np.sqrt(data_original)

# D'agostino K^2
stat_orig, p_orig = stats.normaltest(data_original)
stat_boxcox, p_boxcox = stats.normaltest(price_boxcox)
stat_log, p_log = stats.normaltest(price_log)
stat_sqrt, p_sqrt = stats.normaltest(price_sqrt)

# p-değerine göre durum belirleme: p < 0.5 ise "Başarılı", değilse "Başarısız"
def durum_belirle(p):
    return "Başarılı" if p < 0.5 else "Başarısız"

# Sonuçların tabloya dökülmesi
transformation_df = pd.DataFrame({
    "Dönüşüm": ["Orijinal", "Boxcox", "Log", "Square Root"],
    "p-değeri": [p_orig, p_boxcox, p_log, p_sqrt],
    "Durum": [durum_belirle(p_orig), durum_belirle(p_boxcox), durum_belirle(p_log), durum_belirle(p_sqrt)]
})

print("\nNormalizasyon ve Dönüşüm Karşılaştırması:")
print(transformation_df)

plt.figure(figsize=(6,4))
plt.axis('off')
table = plt.table(cellText=transformation_df.values, colLabels=transformation_df.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.title("Dönüşüm Karşılaştırması")
plt.show()
