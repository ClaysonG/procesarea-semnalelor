import numpy as np
import matplotlib.pyplot as plt

# Generez semnalul sinusoidal
A = 1.0
f = 10.0
phi = 0
t = np.linspace(0, 1, 1000)
x1 = A * np.sin(2*np.pi*f*t + phi)

# Generez semnalul cosinus
x2 = A * np.cos(2*np.pi*f*t + phi - np.pi/2)

# Afisez semnalele
_, axs = plt.subplots(2)
axs[0].plot(t, x1)
axs[0].set_title('Semnal sinusoidal')
axs[1].plot(t, x2)
axs[1].set_title('Semnal cosinus')
plt.tight_layout()
plt.show()
