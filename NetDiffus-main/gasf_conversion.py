import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from skimage.transform import resize
import sys
print(sys.executable)

# Percorsi dei file
file_path = "Mirage2019/Mirage-2019.parquet"  # File Parquet di input
save_dir = "Mirage2019/output"  # Directory per salvare le immagini GASF
n = 1  # Numero di righe (serie temporali) da processare per ogni label

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

    all_series.append(data)

# Converte le serie in un array numpy
all_series = np.array(all_series)

# Calcolo del minimo e massimo globali su tutte le serie temporali
X_min = np.min(all_series)
X_max = np.max(all_series)

# Funzione per normalizzare una serie temporale
def min_max_normalize(series, X_min, X_max):
    return (series - X_min) / (X_max - X_min)

# Normalizzazione delle serie temporali
normalized_series = [min_max_normalize(series, X_min, X_max) for series in all_series]

# Calcolo del GASF
gasf = GramianAngularField(sample_range=(0, 1), method='summation')
X_gasf = gasf.transform(normalized_series)

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

    # Salva l'immagine nella directory della classe
    filename = f"sample_{index + 1}.png"
    filepath = os.path.join(class_dir, filename)
    plt.imsave(filepath, gasf_img, cmap='viridis')

    print(f"Serie {index} salvata come immagine in {filepath}.")

# Salvataggio di X_min e X_max
min_max_file = os.path.join(save_dir, "min_max_values.txt")
with open(min_max_file, "w") as f:
    f.write(f"X_min: {X_min}\n")
    f.write(f"X_max: {X_max}\n")

print(f"Minimo e massimo salvati in {min_max_file}.")
