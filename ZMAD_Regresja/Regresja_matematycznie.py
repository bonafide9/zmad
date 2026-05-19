import numpy as np

X = np.array([30, 50, 70, 90, 110, 130])
y = np.array([180, 250, 310, 370, 450, 480])

x_mean = np.mean(X) # Wynik: 80.0
y_mean = np.mean(y) # Wynik: 340.0

licznik = np.sum((X - x_mean) * (y - y_mean)) # Nasze zsumowane 21600
mianownik = np.sum((X - x_mean)**2)           # Nasze zsumowane 7000

a = licznik / mianownik
b = y_mean - (a * x_mean)

print(f"Współczynnik kierunkowy (a): {a:.4f}")
print(f"Wyraz wolny (b): {b:.4f}")
print(f"Równanie prostej: y = {a:.4f} * x + {b:.4f}")

x_new = 100
y_pred = a * x_new + b

print(f"\nSzacowana cena dla {x_new} m²: {y_pred:.2f} tys. zł")