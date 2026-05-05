import numpy as np
from scipy import stats

# ==========================================
# Dane
# ==========================================
X1 = np.array([2, 3, 2, 4, 3])
X2 = np.array([2, 3, 2, 4, 8])

# ==========================================
# Parametr testu
# ==========================================
alpha = 0.05  # poziom istotności

# ==========================================
# Statystyki
# ==========================================
n1, n2 = len(X1), len(X2)

mu1, mu2 = np.mean(X1), np.mean(X2)
var1 = np.var(X1, ddof=1)
var2 = np.var(X2, ddof=1)

print("=== STATYSTYKI ===")
print(f"Średnia X1: {mu1:.3f}")
print(f"Średnia X2: {mu2:.3f}")
print(f"Wariancja X1: {var1:.3f}")
print(f"Wariancja X2: {var2:.3f}")

# ==========================================
# Statystyka t (jak w zadaniu)
# ==========================================
T = abs(mu1 - mu2) / np.sqrt((var1 + var2) / 2)

# stopnie swobody
df = n1 + n2 - 2

# próg z rozkładu t (test dwustronny)
t_crit = stats.t.ppf(1 - alpha/2, df)

print("\n=== TEST t-STUDENTA ===")
print(f"T = {T:.3f}")
print(f"t_crit = {t_crit:.3f} (alpha={alpha}, df={df})")

# ==========================================
# Decyzja
# ==========================================
print("\n=== DECYZJA ===")
if T > t_crit:
    print("Dryf wykryty (różnica istotna statystycznie)")
else:
    print("Brak dryfu")

# ==========================================
# (Opcjonalnie) klasyczny test t z SciPy
# ==========================================
t_stat, p_value = stats.ttest_ind(X1, X2, equal_var=True)

print("\n=== WERYFIKACJA (scipy.stats.ttest_ind) ===")
print(f"t_stat = {t_stat:.3f}")
print(f"p_value = {p_value:.5f}")

if p_value < alpha:
    print("Dryf wykryty (p-value < alpha)")
else:
    print("Brak dryfu")