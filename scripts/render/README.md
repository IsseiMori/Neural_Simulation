## Rendering the predicted result
```bash
 python render_predicted.py --data=rollout_*.pkl --restpos --mpm or --flex
```
For the data with resting positions, use `--restpos` to render the particle positions. Without this option, it will render the resting positions instead. 


## Rendering the raw numpy dataset
```bash
 python render_npy.py --data=rollout_*.pkl --restpos --mpm or --flex
```
Specify `--mpm` or `--flex` to indicate the dataset type