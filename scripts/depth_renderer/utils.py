import numpy as np
import math

from PIL import Image as im


'''
Save the depth image
Normalize the depth so that is visible
Used in the point rasterizer and the marching cube
'''
def save_depth_image(depth_data, file_name):
    _min = np.amin(depth_data[depth_data != 0])
    _max = np.amax(depth_data[depth_data != 0])
    # print(_min)
    # print(_max)
    _min = -0.7
    _max = -0.4
    disp_norm = (depth_data - _min) * 255.0 / (_max - _min)
    disp_norm = np.clip(disp_norm, a_min = 0, a_max = 255)
    disp_norm[depth_data == 0] = 0
    disp_norm = np.uint8(disp_norm)
    data = im.fromarray(disp_norm).convert('RGB')
    data.save(file_name)


def compute_depth_mc(points, rot, trans, show_canvas=False):

    import trimesh
    from tqdm import tqdm
    import pyrender
    from skimage import measure


    # radius = 0.03  # point radius
    # dx = 0.008  # marching cube grid size

    radius = 0.01  # point radius
    dx = 0.016  # marching cube grid size

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
    

    verts, faces, normals, values = measure.marching_cubes(dist, 0)

    base = np.floor(verts).astype(int)
    frac = verts - base

    new_verts = (
        point_field[base[:, 0], base[:, 1], base[:, 2]] * (1 - frac)
        + point_field[base[:, 0] + 1, base[:, 1] + 1, base[:, 2] + 1] * frac
    )


    # Conver to pyrender mesh
    mesh = trimesh.Trimesh(new_verts, faces)
    # mesh.show()

    cam = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0), aspectRatio=1)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    mesh_node = scene.add(mesh)


    mesh_node.translation = trans
    mesh_node.rotation = rot

    nc = pyrender.Node(camera=cam, matrix=np.eye(4))

    scene.add_node(nc)

    if show_canvas:
        v = pyrender.Viewer(scene, use_raymond_lighting=True)  

    r = pyrender.OffscreenRenderer(viewport_width=128, viewport_height=128)
    color, depth = r.render(scene)

    depth = -depth

    return depth

    # save_depth_image(depth, 'depth.png')