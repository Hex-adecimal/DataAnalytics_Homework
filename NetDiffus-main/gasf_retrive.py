import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

# Percorsi dei file
save_dir = "Mirage2019/output"  # Directory delle immagini GASF salvate
min_max_file = os.path.join(save_dir, "min_max_values.txt")  # File con X_min e X_max
reconstructed_dir = "Mirage2019/reconstructed"  # Directory per salvare le serie ricostruite

# Assicura che la directory di output esista
if not os.path.exists(reconstructed_dir):
    os.makedirs(reconstructed_dir)

# Carica X_min e X_max dal file
with open(min_max_file, "r") as f:
    lines = f.readlines()
    X_min = float(lines[0].split(":")[1].strip())
    X_max = float(lines[1].split(":")[1].strip())

# Funzione per denormalizzare una serie temporale
def inverse_min_max_normalize(normalized_series, X_min, X_max):
    return normalized_series * (X_max - X_min) + X_min

# Funzione per invertire il GASF
def inverse_gasf(gasf_image):
    theta = np.arccos(gasf_image)  # Calcolo degli angoli (arcocoseno)
    reconstructed_series = np.cos(theta.sum(axis=0))  # Ricostruzione della serie temporale
    return reconstructed_series

# Itera attraverso le cartelle di classe
for class_name in os.listdir(save_dir):
    class_dir = os.path.join(save_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    reconstructed_file = os.path.join(reconstructed_dir, f"reconstructed_{class_name}.txt")
    
    with open(reconstructed_file, "w") as f_out:
        print(f"Classe: {class_name}")
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)

            # Carica l'immagine GASF
            gasf_img = imread(file_path, as_gray=False)

            # Normalizzazione inversa per riportare i valori nel range [-1, 1]
            gasf_img = (gasf_img - 0.5) * 2

            # Inversione del GASF per ottenere la serie temporale normalizzata
            reconstructed_normalized_series = inverse_gasf(gasf_img)

            # Denormalizzazione della serie temporale
            reconstructed_series = inverse_min_max_normalize(reconstructed_normalized_series, X_min, X_max)

            # Salva la serie temporale nel file
            f_out.write(f"Serie temporale da {file_name}:\n")
            f_out.write(" ".join(map(str, reconstructed_series)) + "\n\n")
        
        print(f"Serie temporali ricostruite salvate in {reconstructed_file}.")