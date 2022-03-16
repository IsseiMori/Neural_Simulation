python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full/data --flex --restpos --num_data=4000 --offset=0 --name=train
python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full/data --flex --restpos --num_data=1000 --offset=4000 --name=test
python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full/data/rollouts --flex --restpos --num_data=10 --offset=0 --name=train
python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full/data/rollouts --flex --restpos --num_data=10 --offset=4000 --name=test
python create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/full/data --flex --restpos


python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/data --mpm --num_data=4000 --offset=0 --name=train
python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/data --mpm --num_data=1000 --offset=4000 --name=test
python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/data/rollouts --mpm --num_data=10 --offset=0 --name=train
python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/data/rollouts --mpm --num_data=10 --offset=4000 --name=test
python create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/data --mpm