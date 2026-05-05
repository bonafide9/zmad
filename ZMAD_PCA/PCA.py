import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Centrowanie danych
X_centered = X - np.mean(X, axis=0)

# 2. Przekształcenie PCA
pca = PCA(n_components=3)
pca.fit(X_centered)

# Macierz przekształcająca K (Loadingi)
K = pca.components_.T

# Nowe dane Y = X * K
Y = np.dot(X_centered, K)

# 3. Analiza i Wizualizacja
vars_x = np.var(X_centered, axis=0) # Wariancja oryginalnych cech
vars_y = np.var(Y, axis=0)          # Wariancja nowych cech (PCA)

# --- NOWA SEKCJA: WYPISYWANIE WYNIKÓW ---
print("Wariancje cech oryginalnych (x1, x2, x3):")
for i, v in enumerate(vars_x, 1):
    print(f"  Cecha x{i}: {v:.4f}")

print("\nWariancje nowych cech po PCA (y1, y2, y3):")
for i, v in enumerate(vars_y, 1):
    print(f"  Cecha y{i}: {v:.4f}")

print(f"\nSuma wariancji (X): {np.sum(vars_x):.4f}")
print(f"Suma wariancji (Y): {np.sum(vars_y):.4f}")
# ---------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Wykres wariancji
ax1.bar(['$x_1$', '$x_2$', '$x_3$'], vars_x, color='blue', alpha=0.6, label='Stare cechy')
ax1.bar(['$y_1$', '$y_2$', '$y_3$'], vars_y, color='green', alpha=0.6, label='Nowe cechy (PCA)')
ax1.set_title(f'Suma wariancji: {np.sum(vars_x):.2f}')
ax1.legend()

# Wykres instancji w nowej przestrzeni (y1, y2)
ax2.scatter(Y[:, 0], Y[:, 1], c='green', s=100, edgecolor='black')
for i, txt in enumerate(range(1, 7)):
    ax2.annotate(f'Instancja {txt}', (Y[i, 0], Y[i, 1]+0.02))

ax2.set_title('Instancje po redukcji wymiarowości')
ax2.set_xlabel('Nowa Cecha 1 ($y_1$)')
ax2.set_ylabel('Nowa Cecha 2 ($y_2$)')
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.show()