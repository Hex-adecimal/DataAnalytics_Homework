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


#### Esecuzione dello script per avviare il training
Portarsi all'interno della cartella NetDiffus-main
```bash
python3 scripts/image_train.py --data_dir dataset/output/ --image_size 128 --num_channels 64 --num_res_blocks 2 --diffusion_steps 1000 --noise_schedule cosine --learn_sigma True --class_cond True --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 4
```
Nota: nel file NetDiffus-main/scripts/script_util.py c'è la macro NUM_CLASSES impostata a 40; per lovare su più classi modificare questo parametro.

### Link utili
- [Mirage website](https://traffic.comics.unina.it/mirage/index.html)
- [Encoding Time Series as Images](https://medium.com/analytics-vidhya/encoding-time-series-as-images-b043becbdbf3)
