rem python flex_depth_render.py --scene=RiceGrip --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/depth
rem python flex_depth_render.py --scene=PressDown --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/PressDown/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/PressDown/FLEX/depth

rem python mpm_depth_render.py --scene=RiceGrip
rem python mpm_depth_render.py --scene=CompressTube


python flex_depth_render.py --scene=RiceGrip --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/depth

python flex_depth_render.py --scene=RiceGrip --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/depth --predicted



python mpm_depth_render.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/finetune/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/finetune/depth

python bf_loss.py \
--flex_data=../../../tmp/BruteForce/RiceGrip/FLEX/raw \
--flex_depth=../../../tmp/BruteForce/RiceGrip/FLEX/depth \
--mpm_data=../../../tmp/BruteForce/RiceGrip/MPM/raw \
--mpm_depth=../../../tmp/BruteForce/RiceGrip/MPM/depth \
--out=../../../tmp/BruteForce/RiceGrip/sysid


python mpm_depth_render.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/MPM/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/MPM/depth