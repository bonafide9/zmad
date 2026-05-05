import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

# 1. Macierz X: 6 instancji (wiersze), 3 cechy (kolumny)
data = [
    [2.5, 2.4, 0.5], # Instancja 1
    [0.5, 0.7, 0.2], # Instancja 2
    [2.2, 2.9, 0.8], # Instancja 3
    [1.9, 2.2, 0.4], # Instancja 4
    [3.1, 3.0, 1.1], # Instancja 5
    [2.3, 2.7, 0.6]  # Instancja 6
]
df = pd.DataFrame(data, columns=['Cecha_x1', 'Cecha_x2', 'Cecha_x3'])
X = df.values

# Centrowanie danych i zapamiętanie średniej
mean_X = np.mean(X, axis=0)
X_centered = X - mean_X

# 2. Przekształcenie PCA
pca = PCA(n_components=3)
pca.fit(X_centered)

# Macierz przekształcająca K (Loadingi)
K = pca.components_.T

# Nowe dane Y = X * K
Y = np.dot(X_centered, K)

# 3. ODWROTNA TRANSFORMACJA PCA Z USUNIĘCIEM y3

# Krok 3a: Redukcja wymiarowości - usuwamy y3 (zerujemy trzecią kolumnę)
# pca w sklearn domyślnie sortuje komponenty od największej do najmniejszej wariancji,
# więc y3 (indeks 2) ma najmniejszą wariancję.
Y_reduced = Y.copy()
Y_reduced[:, 2] = 0

# Krok 3b: Odwrotne przekształcenie - mnożymy przez K transponowane
X_reconstructed_centered = np.dot(Y_reduced, K.T)

# Krok 3c: Odtworzenie oryginalnej macierzy poprzez dodanie wcześniej odjętej średniej
X_reconstructed = X_reconstructed_centered + mean_X

# 4. Wyświetlenie wyników w celu porównania
np.set_printoptions(precision=4, suppress=True) # dla czytelności wypisywania

print("Oryginalna macierz X:")
print(X)

print("\nOdtworzona macierz X (po usunięciu y3):")
print(X_reconstructed)

print("\nRóżnica między oryginalną a odtworzoną macierzą (utrata informacji):")
print(np.abs(X - X_reconstructed))

print(f"\nSuma wariancji utraconej (wariancja usuniętej cechy y3): {np.var(Y[:, 2]):.4f}")