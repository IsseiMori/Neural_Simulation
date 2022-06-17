rem python inference.py \
rem     --data_path=../../tmp/Grid/RiceGrip/FLEX/data \
rem     --metadata_path=../../tmp/Finetune/RiceGrip/FLEX/full/data \
rem     --model=../../tmp/Finetune/RiceGrip/FLEX/full/models/model_335000.pth \
rem     --output_path=../../tmp/Finetune/RiceGrip/FLEX/full/grid  \
rem     --dim=6

rem python inference.py \
rem     --data_path=../../tmp/Grid/RiceGrip/FLEX/data \
rem     --metadata_path=../../tmp/Finetune/RiceGrip/FLEX/partial/data \
rem     --model=../../tmp/Finetune/RiceGrip/FLEX/partial/models/model_370000.pth \
rem     --output_path=../../tmp/Finetune/RiceGrip/FLEX/partial/grid  \
rem     --dim=6


rem scp -r ../../tmp/groop/data imori@bayes.ucsd.edu:/home/imori/NNSim/Neural_Simulation/tmp/groop

rem scp -r imori@bayes.ucsd.edu:/home/imori/NNSim/Neural_Simulation/tmp/groop/models ../../tmp/groop/remote
rem scp -r imori@bayes.ucsd.edu:/home/imori/NNSim/Neural_Simulation/tmp/groop/rollouts ../../tmp/groop/remote

scp -r imori@bayes.ucsd.edu:/home/imori/NNSim/Neural_Simulation/tmp/groop/energy_conservation/models ../../tmp/groop/energy_conservation
scp -r imori@bayes.ucsd.edu:/home/imori/NNSim/Neural_Simulation/tmp/groop/energy_conservation/rollouts ../../tmp/groop/energy_conservation

rem bash ./learning_to_simulate/download_dataset.sh Goop-3D /home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/groop3d/data


rem python -m learning_to_simulate.train \
rem --data_path=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/groop3d/data \
rem --model_path=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/groop3d/models

rem python train.py \
rem --data_path=../../tmp/groop3d/data \
rem --model_path=../../tmp/groop3d/models  \
rem --output_path=../../tmp/groop3d/rollouts \
rem --eval_steps=10000 \
rem --num_eval_steps=1000 \
rem --save_steps=1000 \
rem --rollout_steps=100000 \
rem --num_rollouts=5 \
rem --num_steps=3000000 \
rem --dim=3 \
rem --batch=1

rem python -m learning_to_simulate.render_rollout \
rem --rollout_path=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/groop/energy_conservation/rollouts/train_200000/rollout_0.pkl