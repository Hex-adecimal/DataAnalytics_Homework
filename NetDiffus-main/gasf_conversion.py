import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import imageio.v2 as imageio
import sys
print(sys.executable)

# Percorsi dei file
file_path = "Mirage2019/Mirage-2019.parquet"  # File Parquet di input
save_dir = "Mirage2019/output"  # Directory per salvare le immagini GASF
series_csv_path = "Mirage2019/series_temporali.csv"  # File CSV per salvare serie temporali
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
series_records = []  # Lista per memorizzare le serie temporali e i label

for index, row in df_grouped.iterrows():
    # Estrae i dati numerici (esclude l'ultima colonna)
    data = row.iloc[0]

    # Nel caso i dati siano inferiori a 10, inseriamo dei dummy
    if len(data) < 10:
        missing_elements = 10 - len(data)
        data = np.append(data, [0] * missing_elements)

    all_series.append(data)
    series_records.append({'serie_temporali': data.tolist(), 'label': row['label']})  # Memorizza i dati

# Salva le serie temporali e le etichette in un file CSV
series_df = pd.DataFrame(series_records)
series_df.to_csv(series_csv_path, index=False)
print(f"Serie temporali salvate in {series_csv_path}.")

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
gasf = GramianAngularField(sample_range=None, method='summation')
X_gasf = gasf.transform(normalized_series)

# Salvataggio delle immagini per ogni classe
for idx, (index, row) in enumerate(df_grouped.iterrows()):
    # Estrae il valore della colonna 'label' come class_name
    class_name = str(row['label'])

    # Crea la directory per la classe se non esiste
    class_dir = os.path.join(save_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Estrae l'immagine GASF corrispondente e porta in [0,1]
    gasf_img = X_gasf[idx] * 0.5 + 0.5

    # Converti in uint8 SUPPORTATO DA IMAGEIO
    #gasf_img_uint8 = (gasf_img * 255).astype(np.uint8)

    # Salva l'immagine in scala di grigi
    filename = f"sample_{index + 1}.npz"
    filepath = os.path.join(class_dir, filename)
    #plt.imsave(filepath, gasf_img, cmap='gray', vmin=0, vmax=1)
    imageio.imwrite(filepath, gasf_img, format=None)

    print(f"Serie {index} salvata come immagine in {filepath}.")

# Salvataggio di X_min e X_max
min_max_file = os.path.join(save_dir, "min_max_values.txt")
with open(min_max_file, "w") as f:
    f.write(f"X_min: {X_min}\n")
    f.write(f"X_max: {X_max}\n")

print(f"Minimo e massimo salvati in {min_max_file}.")
