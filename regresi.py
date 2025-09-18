import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# 1. Baca data
data = pd.read_csv("data.csv")
print("Data Awal:")
print(data.head())

# -----------------------------
# REGRESI LINEAR SEDERHANA
# -----------------------------
print("\n=== Regresi Linear Sederhana (Hours_Studied -> Exam_Score) ===")

X = data[["Hours_Studied"]]   # variabel independen
y = data["Exam_Score"]        # variabel dependen

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

y_pred_simple = model_simple.predict(X_test)

print("Koefisien (slope):", model_simple.coef_[0])
print("Intercept:", model_simple.intercept_)
print("R2 Score:", r2_score(y_test, y_pred_simple))

plt.scatter(X, y, color="blue", label="Data Asli")
plt.plot(X, model_simple.predict(X), color="red", label="Garis Regresi")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Regresi Linear Sederhana")
plt.legend()
plt.show()

# -----------------------------
# REGRESI LINEAR BERGANDA
# -----------------------------
print("\n=== Regresi Linear Berganda (Hours_Studied, IQ, Attendance -> Exam_Score) ===")

X_multi = data[["Hours_Studied", "IQ", "Attendance"]]
y_multi = data["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)

print("Koefisien:", model_multi.coef_)
print("Intercept:", model_multi.intercept_)
print("R2 Score:", r2_score(y_test, y_pred_multi))
print("MAE:", mean_absolute_error(y_test, y_pred_multi))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_multi)))
