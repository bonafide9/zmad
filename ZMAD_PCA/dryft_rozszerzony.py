import numpy as np
from scipy import stats

# ==========================================
# Dane i statystyki podstawowe
# ==========================================
X1 = np.array([2, 3, 2, 4, 3])
X2 = np.array([2, 3, 2, 4, 8])
alpha = 0.05

n1, n2 = len(X1), len(X2)
mu1, mu2 = np.mean(X1), np.mean(X2)
var1 = np.var(X1, ddof=1)
var2 = np.var(X2, ddof=1)

print("=== KLASYCZNY TEST t-STUDENTA (Zgodnie z PDF) ===")

# ==========================================
# 1. Wariancja połączona (Wzór 7)
# ==========================================
# Wzór: s_p^2 = [ (n1 - 1)*s1^2 + (n2 - 1)*s2^2 ] / (n1 + n2 - 2)
sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

# ==========================================
# 2. Statystyka testowa t (Wzór 6)
# ==========================================
# Wzór: t = |mu1 - mu2| / sqrt( sp2 * (1/n1 + 1/n2) )
t_stat = abs(mu1 - mu2) / np.sqrt(sp2 * (1/n1 + 1/n2))

# ==========================================
# 3. Stopnie swobody i próg decyzyjny (Wzory 8 i 9)
# ==========================================
df = n1 + n2 - 2
t_crit = stats.t.ppf(1 - alpha/2, df)

print(f"Wariancja połączona (sp2) = {sp2:.3f}")
print(f"t = {t_stat:.3f}")
print(f"t_crit = {t_crit:.3f}")

# ==========================================
# 4. Decyzja
# ==========================================
print("\n=== DECYZJA ===")
if t_stat > t_crit:
    print("Dryf wykryty (|t| > t_crit)")
else:
    print("Brak dryfu (|t| <= t_crit)")