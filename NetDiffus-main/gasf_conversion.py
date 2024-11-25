import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from skimage.transform import resize

# Percorsi dei file
file_path = "Mirage2019/Mirage-2019.parquet"  # File Parquet di input
save_dir = "Mirage2019/output"  # Directory per salvare le immagini GASF
n = 100  # Numero di righe (serie temporali) da processare per ogni label

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

    # Nel caso i dati sono inferiori di 10, inseriamo dei dummy
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
gasf = GramianAngularField(sample_range=(0, 1), method='summation')
X_gasf = gasf.transform(X)

# Salvataggio delle immagini
for idx, (index, row) in enumerate(df_grouped.iterrows()):
    # Estrae l'ultima colonna come etichetta
    label = row.iloc[-1]
    if not isinstance(label, str):
        label = "Unknown"

    # Estrae l'immagine GASF corrispondente
    gasf_img = X_gasf[idx] * 0.5 + 0.5
    gamma = 0.25
    gasf_img = np.power(gasf_img, gamma)

    # Ridimensiona l'immagine a 128x128
    gasf_img_resized = resize(gasf_img, (128, 128), anti_aliasing=True)

    # Salva l'immagine come PNG
    filename = f"{label}_series_{index+1}.png"
    plt.imsave(os.path.join(save_dir, filename), gasf_img_resized, cmap='viridis')

    print(f"Serie {index} salvata come immagine in {save_dir}/{filename}.")
