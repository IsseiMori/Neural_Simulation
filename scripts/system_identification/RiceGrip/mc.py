import open3d
import trimesh
import numpy as np
from tqdm import tqdm
from pyglet import gl

import pyrender

radius = 0.01  # point radius
dx = 0.004  # marching cube grid size

points = np.random.rand(10,3)
bbox = np.array([points.min(0) - radius * 1.1, points.max(0) + radius * 1.1])

dim = np.ceil((bbox[1] - bbox[0]) / dx)
bbox = np.array([bbox[0], bbox[0] + dim * dx])

dim = dim.astype(int)

point_field = np.stack(
    np.meshgrid(
        np.linspace(bbox[0, 0], bbox[1, 0], dim[0]),
        np.linspace(bbox[0, 1], bbox[1, 1], dim[1]),
        np.linspace(bbox[0, 2], bbox[1, 2], dim[2]),
        indexing="ij",
    ),
    -1,
)

dist = np.ones(point_field.shape[:3]) * 1e6


for p in tqdm(points):
    d = np.linalg.norm(point_field - p, axis=-1)
    dist = np.minimum(dist, d)

dist -= radius


from skimage import measure

verts, faces, normals, values = measure.marching_cubes(dist, 0)

base = np.floor(verts).astype(int)
frac = verts - base

new_verts = (
    point_field[base[:, 0], base[:, 1], base[:, 2]] * (1 - frac)
    + point_field[base[:, 0] + 1, base[:, 1] + 1, base[:, 2] + 1] * frac
)

# print(values)
# print(values.shape)


mesh = trimesh.Trimesh(new_verts, faces)mesh.show()


# mesh = pyrender.Mesh.from_trimesh(mesh)
# scene = pyrender.Scene()
# scene.add(mesh)
# # pyrender.Viewer(scene, use_raymond_lighting=True)

# r = pyrender.OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
# color, depth = r.render(scene)

# print(color)


# scene = mesh.scene()


# png = scene.save_image(resolution=[1920, 1080], visible=False)

# with open('images_mpm/render.png', 'wb') as f:
#     f.write(png)
#     f.close()