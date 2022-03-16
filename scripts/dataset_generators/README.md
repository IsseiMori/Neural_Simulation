# Create tfrecord dataset from raw npy data


## Create tfrecord dataset from raw npy data

```
python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/full/data --simulator=flex --restpos --num_data=10 --offset=0 --name=train
```
- `--restpos` : use this option to keep 3 dim restpos in FLEX data or add extra 3 dim in MPM data


## Create metadata from raw npy data

```
python create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/full/data --simulator=flex --restpos --num_data=10 --offset=0
```

- `--restpos` : use this option to keep 3 dim restpos in FLEX data or add extra 3 dim in MPM data