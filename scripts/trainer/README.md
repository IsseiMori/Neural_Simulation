## Training GNN
```bash
python train.py \
    --data_path=../../tmp/FLEX_RiceGrip/data \
    --model_path=../../tmp/FLEX_RiceGrip/models \
    --output_path=../../tmp/FLEX_RiceGrip/rollouts \
    --eval_steps=100000 \
    --num_eval_steps=1000 \
    --save_steps=100 \
    --rollout_steps=500 \
    --dim=6
```
For the data with resting positions, use `-dim=6`.