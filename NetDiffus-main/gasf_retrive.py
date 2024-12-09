import os
import numpy as np
import imageio.v2 as imageio

# Percorsi
save_dir = "Mirage2019/output"  # Directory con le immagini GASF e il file min_max_values.txt
min_max_file = os.path.join(save_dir, "min_max_values.txt")  # File con X_min e X_max
output_file = "Mirage2019/reconstructed_series_all.txt"  # File unico per tutte le serie ricostruite

# Carica X_min e X_max
with open(min_max_file, "r") as f:
    lines = f.readlines()
    X_min = float(lines[0].split(":")[1].strip())
    X_max = float(lines[1].split(":")[1].strip())

# Funzione per denormalizzare una serie temporale
def inverse_min_max_normalize(normalized_series, X_min, X_max):
    return np.round(normalized_series * (X_max - X_min) + X_min)

with open(output_file, "w") as f_out:
    # Scorri tutte le sottodirectory corrispondenti alle classi
    for class_name in os.listdir(save_dir):
        class_dir = os.path.join(save_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Scorri tutti i file PNG nella directory della classe
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(".npz"):
                file_path = os.path.join(class_dir, file_name)

                # Carica l'immagine GASF in scala di grigi
                gasf_img = imageio.imread(file_path)

                # Riporta l'immagine da [0,1] a [-1,1]
                gasf_img = (gasf_img - 0.5) * 2

                # Inversione del GASF
                # Dalla diagonale otteniamo GAF(i,i) = cos(2*φ_i), quindi φ_i = arccos(GAF(i,i))/2
                diag_values = np.diag(gasf_img)
                phi = np.arccos(diag_values) / 2.0

                # Ricostruzione della serie normalizzata: X[i] = cos(φ_i)
                reconstructed_normalized_series = np.cos(phi)

                # Denormalizzazione della serie temporale
                reconstructed_series = inverse_min_max_normalize(reconstructed_normalized_series, X_min, X_max)

                # Scrittura della serie nel file di output
                f_out.write(f"Classe: {class_name}, File: {file_name}\n")
                f_out.write(" ".join(map(str, reconstructed_series)) + "\n\n")

print(f"Serie temporali ricostruite salvate in {output_file}.")