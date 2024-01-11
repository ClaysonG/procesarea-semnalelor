import numpy as np
import matplotlib.pyplot as plt

# Definirea parametrilor distribuției unidimensionale
media_unidimensionala = 0
deviatia_standard_unidimensionala = 1
numar_eșantioane = 1000

# Generarea eșantioanelor din distribuția unidimensională
eșantioane_unidimensionale = np.random.normal(
    media_unidimensionala, deviatia_standard_unidimensionala, numar_eșantioane)

# Afișarea distribuției unidimensionale pe un grafic
plt.hist(eșantioane_unidimensionale, bins=30,
         density=True, alpha=0.7, color='blue')
plt.title('Distribuție Gaussiană Unidimensională')
plt.xlabel('Valori')
plt.ylabel('Probabilitate')
plt.show()

# Definirea parametrilor distribuției bidimensionale
media_bidimensionala = np.array([0, 0])
matrice_covarianta_bidimensionala = np.array([[1, 0.5], [0.5, 1]])
numar_eșantioane = 1000

# Generarea eșantioanelor din distribuția bidimensională
eșantioane_bidimensionale = np.random.multivariate_normal(
    media_bidimensionala, matrice_covarianta_bidimensionala, numar_eșantioane)

# Afișarea distribuției bidimensionale pe un grafic
plt.scatter(eșantioane_bidimensionale[:, 0],
            eșantioane_bidimensionale[:, 1], color='green', alpha=0.5)
plt.title('Distribuție Gaussiană Bidimensională')
plt.xlabel('Valoare X')
plt.ylabel('Valoare Y')
plt.show()
