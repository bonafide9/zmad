import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Wczytanie danych1
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# 2. Standaryzacja
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Informacja o wariancji
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Suma wyjaśnionej wariancji:", np.sum(pca.explained_variance_ratio_))

# 5. Wagi cech składowych
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=feature_names)

print("\nWagi cech:")
print(loadings)

# 6. Wykres PCA
plt.figure(figsize=(8, 6))
for class_value, class_name in enumerate(target_names):
    plt.scatter(X_pca[y == class_value, 0], X_pca[y == class_value, 1], label=class_name, alpha=0.7)
plt.xlabel('Pierwsza składowa główna (PC1)')
plt.ylabel('Druga składowa główna (PC2)')
plt.title('PCA - Wizualizacja danych')
plt.legend()
plt.grid(True)
plt.show()

# 7. Najważniejsze cechy dla PC1
print('\nCechy o największym wpływie na PC1:')
print(loadings['PC1'].abs().sort_values(ascending=False))