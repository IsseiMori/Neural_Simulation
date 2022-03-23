rem python flex_depth_render.py --scene=RiceGrip --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/depth
rem python flex_depth_render.py --scene=PressDown --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/PressDown/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/PressDown/FLEX/depth

rem python mpm_depth_render.py --scene=RiceGrip
rem python mpm_depth_render.py --scene=CompressTube


python flex_depth_render.py --scene=RiceGrip --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/depth

python flex_depth_render.py --scene=RiceGrip --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/depth --predicted
