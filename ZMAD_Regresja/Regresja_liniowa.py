import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dane treningowe (x – powierzchnia [m²], y – cena [tys. zł])
X = np.array([30, 50, 70, 90, 110, 130]).reshape(-1, 1)
y = np.array([180, 250, 310, 370, 450, 480])

# Utworzenie i dopasowanie modelu regresji liniowej
model = LinearRegression()
model.fit(X, y)

# Wyświetlenie współczynników
print("Współczynnik kierunkowy (a):", model.coef_[0])
print("Wyraz wolny (b):", model.intercept_)

# Predykcja dla domu o powierzchni 100 m²
x_new = np.array([[100]])
y_pred = model.predict(x_new)
print(f"Szacowana cena dla 100 m²: {y_pred[0]:.2f} tys. zł")

# Wizualizacja
plt.scatter(X, y, color='blue', label='Dane uczące')
plt.plot(X, model.predict(X), color='red', label='Regresja liniowa')
plt.scatter(x_new, y_pred, color='green', s=100, label='Nowy punkt (100 m²)')
plt.xlabel("Powierzchnia [m²]")
plt.ylabel("Cena [tys. zł]")
plt.title("Prosty model regresji liniowej")
plt.legend()
plt.show()