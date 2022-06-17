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



rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/data --flex --restpos --num_data=125 --offset=0 --name=grid



rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/data --flex --restpos --num_data=1 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/data --flex --restpos --num_data=1 --offset=0 --name=test
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/data/rollouts --flex --restpos --num_data=1 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/data/rollouts --flex --restpos --num_data=1 --offset=0 --name=test
rem python create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/data --flex --restpos --num_data=1


rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMulti/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/data --flex --restpos --num_data=1 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMulti/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/data --flex --restpos --num_data=1 --offset=0 --name=test
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMulti/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/data/rollouts --flex --restpos --num_data=1 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMulti/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/data/rollouts --flex --restpos --num_data=1 --offset=0 --name=test
rem python create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMulti/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMultiOverfit/data --flex --restpos --num_data=1


rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Grid/RiceGrip/FLEX/data --flex --restpos --num_data=125 --offset=0 --name=test


rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/data --flex --restpos --num_data=4000 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/data --flex --restpos --num_data=500 --offset=4000 --name=test
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/data/rollouts --flex --restpos --num_data=10 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/data/rollouts --flex --restpos --num_data=10 --offset=4000 --name=test
rem python create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandom/FLEX/data --flex --restpos


rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/data --flex --restpos --num_data=4000 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/data --flex --restpos --num_data=500 --offset=4000 --name=test
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/data/rollouts --flex --restpos --num_data=10 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/data/rollouts --flex --restpos --num_data=10 --offset=4000 --name=test
rem python create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripRandomFull/FLEX/data --flex --restpos


rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/mat/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/mat/data --flex --restpos --num_data=4000 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/mat/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/mat/data --flex --restpos --num_data=500 --offset=4000 --name=test
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/mat/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/mat/data/rollouts --flex --restpos --num_data=10 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/mat/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/mat/data/rollouts --flex --restpos --num_data=10 --offset=4000 --name=test
rem python create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/mat/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/FLEX/mat/data --flex --restpos


rem python create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMulti/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGripMulti/data --flex --restpos

rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/Finetune/RiceGrip/MPM/full/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/test/data --mpm --num_data=1 --offset=0 --name=train


rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/data --mpm --num_data=4000 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/data --mpm --num_data=500 --offset=4000 --name=test
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/data/rollouts --mpm --num_data=10 --offset=0 --name=train
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/data/rollouts --mpm --num_data=10 --offset=4000 --name=test
rem python create_metadata.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/data --mpm --num_data=4000

rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/reduced/data --mpm --num_data=4000 --offset=0 --name=train --reduced=1060
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/reduced/data --mpm --num_data=500 --offset=4000 --name=test --reduced=1060
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/reduced/data/rollouts --mpm --num_data=10 --offset=0 --name=train --reduced=1060
rem python create_tfrecord.py --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/raw --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/MPM/reduced/data/rollouts --mpm --num_data=10 --offset=4000 --name=test --reduced=1060


rem python create_tfrecord_multi.py \
rem --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/extra_features/raw \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/extra_features/data \
rem --num_data=100 --offset=0 --name=train

rem python create_tfrecord_multi.py \
rem --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/extra_features/raw \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/extra_features/data \
rem --num_data=20 --offset=100 --name=test


rem python create_tfrecord_multi.py \
rem --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/extra_features/raw \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/extra_features/data/rollouts \
rem --num_data=2 --offset=0 --name=train

rem python create_tfrecord_multi.py \
rem --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/extra_features/raw \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/extra_features/data/rollouts \
rem --num_data=2 --offset=100 --name=test

rem python create_metadata_multi.py \
rem --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/extra_features/raw \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/extra_features/data \
rem --mpm --num_data=100


rem python create_tfrecord_multi.py \
rem --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Plastic/raw \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Plastic/data \
rem --num_data=250 --offset=0 --name=train

rem python create_tfrecord_multi.py \
rem --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Plastic/raw \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Plastic/data \
rem --num_data=50 --offset=250 --name=test

rem python create_tfrecord_multi.py \
rem --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Plastic/raw \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Plastic/data/rollouts \
rem --num_data=2 --offset=0 --name=train

rem python create_tfrecord_multi.py \
rem --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Plastic/raw \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Plastic/data/rollouts \
rem --num_data=2 --offset=250 --name=test

rem python create_metadata_multi.py \
rem --data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Plastic/raw \
rem --out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Plastic/data \
rem --mpm --num_data=250



python create_tfrecord_multi.py \
--data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Elastic/raw \
--out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Elastic/data \
--num_data=250 --offset=0 --name=train

python create_tfrecord_multi.py \
--data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Elastic/raw \
--out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Elastic/data \
--num_data=50 --offset=250 --name=test

python create_tfrecord_multi.py \
--data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Elastic/raw \
--out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Elastic/data/rollouts \
--num_data=2 --offset=0 --name=train

python create_tfrecord_multi.py \
--data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Elastic/raw \
--out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Elastic/data/rollouts \
--num_data=2 --offset=250 --name=test

python create_metadata_multi.py \
--data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Elastic/raw \
--out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/Elastic/data \
--mpm --num_data=250



python create_tfrecord_multi.py \
--data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/elastoplastic/raw \
--out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/elastoplastic/data \
--num_data=250 --offset=0 --name=train

python create_tfrecord_multi.py \
--data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/elastoplastic/raw \
--out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/elastoplastic/data \
--num_data=50 --offset=250 --name=test

python create_tfrecord_multi.py \
--data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/elastoplastic/raw \
--out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/elastoplastic/data/rollouts \
--num_data=2 --offset=0 --name=train

python create_tfrecord_multi.py \
--data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/elastoplastic/raw \
--out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/elastoplastic/data/rollouts \
--num_data=2 --offset=250 --name=test

python create_metadata_multi.py \
--data=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/elastoplastic/raw \
--out=/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/RiceGripMulti/elastoplastic/data \
--mpm --num_data=250