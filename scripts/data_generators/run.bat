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
rem python datagen_mpm.py --video --scene=RiceGrip --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/MPM/raw" --flex="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw"
rem python datagen_mpm.py --video  --scene=PressDown --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/PressDown/MPM/raw" --flex="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/PressDown/FLEX/raw"

rem python datagen_flex.py --scene=RiceGripMulti --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMulti/raw"


rem python datagen_flex.py --video --scene=RiceGrip --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw"


rem python datagen_flex.py --video --scene=RiceGrip --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw"

rem python datagen_mpm.py --video  --scene=RiceGrip --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/finetune/raw" --flex="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw"

rem python datagen_flex.py --video --scene=RiceGrip --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/raw"



rem python datagen_mpm.py --video  --scene=RiceGrip --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/MPM/raw" --flex="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/raw"


rem python datagen_flex.py --scene=RiceGripMulti --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/raw"


rem python datagen_mpm.py --video  --scene=RiceGrip --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/full/mass" --flex="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/full/raw"




rem python datagen_flex.py --scene=RiceGrip --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/mat/raw"



rem python datagen_mpm.py --video  --scene=RiceGrip --out="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/full/mass" --flex="/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/full/raw"


rem python datagen_mpm.py \
rem --scene=RiceGrip0 \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/MPM/raw \
rem --flex=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/raw \
rem --data=00000 \
rem --video

rem python datagen_mpm.py \
rem --scene=RiceGrip1 \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/MPM/raw \
rem --flex=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/raw \
rem --data=00001 \
rem --video

rem python datagen_mpm.py \
rem --scene=RiceGrip2 \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/MPM/raw \
rem --flex=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/raw \
rem --data=00002 \
rem --video

rem python datagen_mpm.py \
rem --scene=RiceGrip3 \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/MPM/raw \
rem --flex=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/raw \
rem --data=00003 \
rem --video

rem python datagen_mpm.py \
rem --scene=RiceGrip4 \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/MPM/raw \
rem --flex=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/raw \
rem --data=00004 \
rem --video



rem python datagen_mpm.py \
rem --scene=RiceGripMulti \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/sysid/raw \
rem --offset=4000 \
rem --n=20 \
rem --video \
rem --ppos

rem python datagen_mpm_multi.py \
rem --scene=RiceGripMulti \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/extra_features/raw \
rem --n=120 \
rem --video

rem python datagen_mpm_multi.py \
rem --scene=RiceGripMulti \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Plastic/raw \
rem --n=500 \
rem --plastic


rem python datagen_mpm_multi.py \
rem --scene=RiceGripMulti \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Elastic/raw \
rem --offset=41 \
rem --n=300

python datagen_mpm_multi.py \
--scene=RiceGripMulti \
--out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/elastoplastic/raw \
--n=300 \
--elastoplastic
