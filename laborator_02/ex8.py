import numpy as np
import matplotlib.pyplot as plt

# Intervalul de valori pentru α in intervalul [-π/2, π/2]
alpha = np.linspace(-np.pi/2, np.pi/2, 100)

# Calculul functiei sin(α) si aproximarea sa sin(α) ≈ α
sin_alpha = np.sin(alpha)
approx_alpha = alpha

# Calculul aproximarii Pade pentru sin(α)
pade_approximation = (alpha - (7/60) * alpha**3) / (1 + (alpha**2)/20)

# Calculul erorii dintre cele doua aproximari
error = np.abs(sin_alpha - approx_alpha)

plt.figure(figsize=(10, 6))

# Afisarea funcției sin(α)
plt.plot(alpha, sin_alpha, label='sin(α)', color='blue')

# Afisarea aproximarii sin(α) ≈ α
plt.plot(alpha, approx_alpha, label='Aproximare: sin(α) ≈ α',
         linestyle='dotted', color='green')

# Afisarea aproximarii Pade pentru sin(α)
plt.plot(alpha, pade_approximation, label='Aproximare Pade',
         linestyle='dotted', color='red')

# Afisarea erorii dintre cele doua aproximari
plt.plot(alpha, error, label='Eroare', color='purple')

plt.title('Aproximare sin(α) și eroare')
plt.xlabel('α')
plt.grid(True)
plt.legend()
plt.show()

# Observatie: Aproximarea sin(α) ≈ α este o aproximare buna in apropiere de origine, dar devine mai inexacta pe masura ce α creste
