### Dipendenze da installare per il framework
pip install:
- pandas
- numpy
- pyts
- matplotlib
- scikit-image
- blobfile
- torch
- torchvision
- mpi4py
- opencv-python


#### Esecuzione dello script per avviare il training
Portarsi all'interno della cartella NetDiffus-main
```bash
python3 scripts/image_train.py --data_dir dataset/output/ --image_size 128 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule cosine --learn_sigma True --class_cond True --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 4
```
Nota: nel file NetDiffus-main/scripts/script_util.py c'è la macro NUM_CLASSES impostata a 40; per lovare su più classi modificare questo parametro.

Se il processo viene killato sostituire i seguenti flag con questi valori:
```bash
python3 ... --num_channels 64 --num_res_blocks 2  ...
```

#### Esecuzione dello script per generare dati
Portarsi all'interno della cartella NetDiffus-main
```bash
python3 scripts/image_sample.py --model_path 128/iterate/df/synth_models/ema_0.9999_000000.pt --image_size 128 --num_channels 128 --num_res_blocks 3 --diffusion_steps 1000 --noise_schedule cosine --learn_sigma True --class_cond True --rescale_learned_sigmas False --rescale_timesteps False
```
Nota: in base al modello cambiare il numero di ema_0.9999_XXXXXX e usare gli sessi num_channels e num_res_blocks del training.

### Link utili
- [Mirage website](https://traffic.comics.unina.it/mirage/index.html)
- [Encoding Time Series as Images](https://medium.com/analytics-vidhya/encoding-time-series-as-images-b043becbdbf3)
