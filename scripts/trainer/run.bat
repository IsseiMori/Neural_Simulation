python inference.py \
    --data_path=../../tmp/Grid/RiceGrip/FLEX/data \
    --metadata_path=../../tmp/Finetune/RiceGrip/FLEX/full/data \
    --model=../../tmp/Finetune/RiceGrip/FLEX/full/models/model_335000.pth \
    --output_path=../../tmp/Finetune/RiceGrip/FLEX/full/grid  \
    --dim=6

python inference.py \
    --data_path=../../tmp/Grid/RiceGrip/FLEX/data \
    --metadata_path=../../tmp/Finetune/RiceGrip/FLEX/partial/data \
    --model=../../tmp/Finetune/RiceGrip/FLEX/partial/models/model_370000.pth \
    --output_path=../../tmp/Finetune/RiceGrip/FLEX/partial/grid  \
    --dim=6


scp -r ../../../tmp/groop/data imori@bayes.ucsd.edu:/home/imori/NNSim/Neural_Simulation/tmp/groop