import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error
from math import sqrt

# Pct a)

# Setare seed pentru reproducibilitate
np.random.seed(42)

# Parametrii seriei de timp
N = 1000
t = np.arange(N)

# Componenta trend (ecuație de gradul 2)
trend = 0.001 * t**2 + 0.1 * t + 10

# Componenta sezon (două frecvențe)
sezon = 5 * np.sin(2 * np.pi * 0.02 * t) + 3 * np.cos(2 * np.pi * 0.05 * t)

# Componenta variabilitate mică (zgomot alb gaussian)
variabilitate = np.random.normal(0, 1, N)

# Seria de timp rezultată
serie_timp = trend + sezon + variabilitate

# Desenarea seriilor de timp și a componentelor individuale
plt.figure(figsize=(12, 6))

plt.subplot(4, 1, 1)
plt.plot(t, serie_timp, label='Seria de timp')
plt.legend()
plt.title('Seria de timp')

plt.subplot(4, 1, 2)
plt.plot(t, trend, label='Trend')
plt.legend()
plt.title('Trend')

plt.subplot(4, 1, 3)
plt.plot(t, sezon, label='Sezon')
plt.legend()
plt.title('Sezon')

plt.subplot(4, 1, 4)
plt.plot(t, variabilitate, label='Variabilitate mică')
plt.legend()
plt.title('Variabilitate mică')

plt.tight_layout()
plt.show()

# Pct b)

# Calcularea autocorelației
autocorrelation = np.correlate(serie_timp, serie_timp, mode='full')

# Normalizarea pentru a obține funcția de autocorelație
autocorrelation /= np.max(autocorrelation)

# Desenarea vectorului de autocorelație
plt.figure(figsize=(12, 6))
plt.plot(autocorrelation, label='Autocorelație')
plt.title('Vectorul de autocorelație')
plt.xlabel('Întârzieri temporale')
plt.ylabel('Valoare de autocorelație normalizată')
plt.legend()
plt.show()

# Pct c)

# Specificați ordinul modelului AR
p = 2

# Ajustați modelul AR
model = sm.tsa.ARIMA(serie_timp, order=(p, 0, 0))
result = model.fit(method='yule_walker')

# Realizați predicții pe baza modelului
predictions = result.predict(start=p, end=len(serie_timp) - 1)

# Afișați seria de timp originală și predicțiile
plt.figure(figsize=(12, 6))
plt.plot(serie_timp, label='Seria de timp originală')
plt.plot(predictions, label='Predicții AR')
plt.title(f'Model AR(p={p}) și Predicții')
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.legend()
plt.show()

# Pct d)

# Interval pentru ordinul modelului AR
min_p, max_p = 1, 10

# Variabile pentru a păstra cel mai bun ordin și cea mai mică eroare
best_p = None
best_rmse = float('inf')

# Iterez prin diferite valori ale ordinului și evaluez performanța
for p in range(min_p, max_p + 1):
    # Ajustez modelul ARMA
    model = sm.tsa.ARIMA(serie_timp, order=(p, 0, 0))
    result = model.fit(method='yule_walker')

    # Realizez predicții pe baza modelului
    predictions = result.predict(start=p, end=len(serie_timp) - 1)

    # Calculez eroarea (Root Mean Squared Error)
    rmse = sqrt(mean_squared_error(serie_timp[p:], predictions))

    # Actualizez cel mai bun ordin dacă am găsit o eroare mai mică
    if rmse < best_rmse:
        best_rmse = rmse
        best_p = p

print(f"Cel mai bun ordin (p): {best_p}")
print(f"Eroarea asociată (RMSE): {best_rmse}")
