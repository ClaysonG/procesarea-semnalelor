# Exercitiul 3
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.ndimage import gaussian_filter

X = misc.face(gray=True)

pixel_noise = 200
noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)

X = misc.face(gray=True)
X_noisy = X + noise
X_denoised = gaussian_filter(X_noisy, sigma=1)

print(f"SNR inainte de filtrare: {np.sum(X**2) / np.sum((X - X_noisy)**2)}")

fig, axs = plt.subplots(1, 3, figsize=(12, 6))
axs[0].imshow(X, cmap=plt.cm.gray)
axs[0].set_title("Imaginea originala")

axs[1].imshow(X_noisy, cmap=plt.cm.gray)
axs[1].set_title("Imaginea cu zgomot")

axs[2].imshow(X_denoised, cmap=plt.cm.gray)
axs[2].set_title("Imaginea fara zgomot")

plt.tight_layout()

plt.show()

print(f"SNR dupa filtrare: {np.sum(X**2) / np.sum((X - X_denoised)**2)}")
