# EDA - Esplorazione del dataset MNIST (versione da Kaggle)
# Dataset: mnist_train.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carico il dataset
df = pd.read_csv("data/mnist_train.csv")

# Controllo dimensioni del dataset
print("Numero di righe e colonne:", df.shape)

# Controllo le prime righe per farmi un’idea
print("\nPrime 5 righe del dataset:")
print(df.head())

# Statistiche descrittive dei pixel
print("\nStatistiche descrittive dei pixel:")
print(df.describe())

# Controllo se ci sono valori nulli
print("\nValori nulli nel dataset:")
print(df.isnull().sum().sum())  # Se è 0, tutto ok

# Distribuzione delle etichette (le cifre da 0 a 9)
plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=df)
plt.title("Distribuzione delle cifre nel dataset")
plt.xlabel("Cifra")
plt.ylabel("Numero di esempi")
plt.show()

# Visualizzo alcune immagini per vedere se i dati hanno senso
def mostra_immagini(df, n=10):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        # i pixel partono dalla seconda colonna (la prima è 'label')
        img = df.iloc[i, 1:].values.reshape(28, 28)
        label = df.iloc[i, 0]
        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.suptitle("Prime immagini del dataset")
    plt.show()

mostra_immagini(df)

# Calcolo l'immagine "media" per ogni cifra
mean_images = df.groupby("label").mean()

plt.figure(figsize=(12, 6))
for i in range(10):
    img = mean_images.iloc[i].values.reshape(28, 28)
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Media: {i}")
    plt.axis('off')
plt.suptitle("Immagine media per ciascuna cifra")
plt.tight_layout()
plt.show()
