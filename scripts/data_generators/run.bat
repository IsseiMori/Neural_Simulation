rem python datagen_flex.py --scene=RiceGrip
rem python datagen_flex.py --scene=PressDown
rem rem rem python datagen_flex.py --scene=BendTube
rem rem rem python datagen_flex.py --scene=CompressTube

rem python datagen_mpm.py --scene=RiceGrip
rem python datagen_mpm.py --scene=PressDown
rem rem python datagen_mpm.py --scene=BendTube
rem rem python datagen_mpm.py --scene=CompressTube


rem python datagen_flex.py --video --scene=RiceGrip --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw"
rem python datagen_flex.py --video --scene=PressDown --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/PressDown/FLEX/raw"
python datagen_mpm.py --video --scene=RiceGrip --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/MPM/raw" --flex="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw"
python datagen_mpm.py --video  --scene=PressDown --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/PressDown/MPM/raw" --flex="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/PressDown/FLEX/raw"