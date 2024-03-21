from PIL import Image, ImageSequence
import numpy as np
import skimage.measure
import skimage.transform
from stl import mesh
import os

def load_gif_as_slices(file_path):
    img = Image.open(file_path)
    slices = [np.array(frame.convert('L')) > 128 for frame in ImageSequence.Iterator(img)]
    return np.array(slices)

def interpolate_slices(slices, output_depth):
    depth, height, width = slices.shape
    interpolated_slices = skimage.transform.resize(slices.astype(float), (output_depth, height, width), order=1, mode='constant', cval=0, clip=True, anti_aliasing=False)
    return (interpolated_slices > 0.5).astype(bool)

def voxels_to_mesh(voxels):
    verts, faces, normals, values = skimage.measure.marching_cubes(voxels, 0)
    return verts, faces

def save_stl(verts, faces, filename):
    mesh_data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_data['vectors'][i][j] = verts[f[j], :]
    my_mesh = mesh.Mesh(mesh_data)
    my_mesh.save(filename)

def main(input_gif_path, loop_num):
    input_gif_path = input_gif_path
    output_stl_path = "C:/Users/chpre/OneDrive/Desktop/results_discussion/cond_guided_stls/"+"temp_output_"+str(loop_num)+".stl"
    output_depth = 64
    
    slices = load_gif_as_slices(input_gif_path)
    interpolated_slices = interpolate_slices(slices, output_depth)
    verts, faces = voxels_to_mesh(interpolated_slices)
    save_stl(verts, faces, output_stl_path)


for k in range (1,58):
    real_image_path = "D:/latest_guided_results/"+str(k)+".gif"
    main(real_image_path, k)
