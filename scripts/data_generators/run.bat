python datagen_flex.py --scene=RiceGrip
python datagen_flex.py --scene=PressDown
rem rem python datagen_flex.py --scene=BendTube
rem rem python datagen_flex.py --scene=CompressTube

python datagen_mpm.py --scene=RiceGrip
python datagen_mpm.py --scene=PressDown
rem rem python datagen_mpm.py --scene=BendTube
rem rem python datagen_mpm.py --scene=CompressTube


python ../dataset_generators/create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial/data --flex --restpos --num_data=4000 --offset=0 --name=train
python ../dataset_generators/create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial/data --flex --restpos --num_data=1000 --offset=4000 --name=test
python ../dataset_generators/create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial/data/rollouts --flex --restpos --num_data=10 --offset=0 --name=train
python ../dataset_generators/create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial/data/rollouts --flex --restpos --num_data=10 --offset=4000 --name=test
python ../dataset_generators/create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/FLEX/partial/data --flex --restpos


python ../dataset_generators/create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/data --mpm --num_data=4000 --offset=0 --name=train
python ../dataset_generators/create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/data --mpm --num_data=1000 --offset=4000 --name=test
python ../dataset_generators/create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/data/rollouts --mpm --num_data=10 --offset=0 --name=train
python ../dataset_generators/create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/data/rollouts --mpm --num_data=10 --offset=4000 --name=test
python ../dataset_generators/create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/PressDown/MPM/partial/data --mpm