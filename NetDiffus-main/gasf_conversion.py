import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt

# Percorsi dei file
file_path = "../dataset/Mirage-2019.parquet"  # File Parquet di input
save_dir = "../dataset/output"  # Directory per salvare le immagini GASF
n = 5  # Numero di righe (serie temporali) da processare per ogni label

# Assicura che la directory di output esista
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Legge il file Parquet
df = pd.read_parquet(file_path)

# Raggruppa per etichetta (ultima colonna) e seleziona i primi n elementi per gruppo
df['label'] = df.iloc[:, -1]  # Aggiunge una colonna 'label' per chiarezza
df_grouped = df.groupby('label').head(n)

# Raccolta di tutte le serie temporali per il fit congiunto
all_series = []

for index, row in df_grouped.iterrows():
    # Estrae i dati numerici (esclude l'ultima colonna)
    data = row.iloc[0]

    # Nel caso i dati siano inferiori a 10, inseriamo dei dummy
    if len(data) < 10:
        missing_elements = 10 - len(data)
        data = np.append(data, [0] * missing_elements)

    # Creazione dei punti per la GASF (qui bin = 1)
    num_of_samples_per_bin = 1
    points = [
        np.sum(data[j * num_of_samples_per_bin:(j + 1) * num_of_samples_per_bin])
        for j in range(len(data) // num_of_samples_per_bin)
    ]
    all_series.append(points)

# Converte le serie in un array numpy
X = np.array(all_series)

# Calcolo del GASF
gasf = GramianAngularField(sample_range=(0,1), method='summation')
X_gasf = gasf.transform(X)

# Salvataggio delle immagini per ogni classe
for idx, (index, row) in enumerate(df_grouped.iterrows()):
    # Estrae il valore della colonna 'label' come class_name
    class_name = str(row['label'])

    # Crea la directory per la classe se non esiste
    class_dir = os.path.join(save_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Estrae l'immagine GASF corrispondente
    gasf_img = X_gasf[idx] * 0.5 + 0.5
    gamma = 0.25
    gasf_img = np.power(gasf_img, gamma)

    # Salva l'immagine nella directory della classe
    filename = f"sample_{index + 1}.png"
    filepath = os.path.join(class_dir, filename)
    plt.imsave(filepath, gasf_img, cmap='viridis')

    print(f"Serie {index} salvata come immagine in {filepath}.")
