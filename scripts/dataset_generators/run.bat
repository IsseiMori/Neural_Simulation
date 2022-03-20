rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial/data --flex --restpos --num_data=4000 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial/data --flex --restpos --num_data=1000 --offset=4000 --name=test
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial/data/rollouts --flex --restpos --num_data=10 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial/data/rollouts --flex --restpos --num_data=10 --offset=4000 --name=test
rem python create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/partial/data --flex --restpos


rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/full/data --mpm --num_data=4000 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/full/data --mpm --num_data=1000 --offset=4000 --name=test
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/full/data/rollouts --mpm --num_data=10 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/full/data/rollouts --mpm --num_data=10 --offset=4000 --name=test

rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/partial/data --mpm --num_data=4000 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/partial/data --mpm --num_data=1000 --offset=4000 --name=test
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/partial/data/rollouts --mpm --num_data=10 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/reduced/partial/data/rollouts --mpm --num_data=10 --offset=4000 --name=test

rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/full/data --mpm --num_data=4000 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/full/data --mpm --num_data=1000 --offset=4000 --name=test
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/full/data/rollouts --mpm --num_data=10 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/full/data/rollouts --mpm --num_data=10 --offset=4000 --name=test

rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/partial/data --mpm --num_data=4000 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/partial/data --mpm --num_data=1000 --offset=4000 --name=test
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/partial/data/rollouts --mpm --num_data=10 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/reduced/partial/data/rollouts --mpm --num_data=10 --offset=4000 --name=test



python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/data --flex --restpos --num_data=125 --offset=0 --name=grid