# Depth Render related


## Render depth maps using Marching Cube and PyRender
This is to create the target depth map dataset for finetuning experiments

```
python mpm_depth_render.py \
--data=raw/*.npy \
--out=tmp/depth \
--views=views.json
--n=10
```

`views.json` should be specified as follows
```
[
    {"view": "top", "rotation": [90, 0, 0], "translation": [0.0, 0.0, -0.6]},
    {"view": "front", "rotation": [0, 0, 0], "translation": [0.0, 0.0, -0.6]},
    {"view": "right", "rotation": [0, 90, 0], "translation": [0.0, 0.0, -0.6]},
    {"view": "left", "rotation": [0, -90, 0], "translation": [0.0, 0.0, -0.6]},
    {"view": "back", "rotation": [0, 180, 0], "translation": [0.0, 0.0, -0.6]}
]
```