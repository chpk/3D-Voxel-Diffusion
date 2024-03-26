import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import os
import gc
from multiprocessing import Pool, freeze_support




def random_rotation_matrix():
    """Generate a random 3D rotation matrix."""
    theta, phi, z = np.random.uniform(0, 2*np.pi, 3)
    r_z = np.array([[np.cos(z), -np.sin(z), 0],
                    [np.sin(z), np.cos(z), 0],
                    [0, 0, 1]])
    r_y = np.array([[np.cos(phi), 0, np.sin(phi)],
                    [0, 1, 0],
                    [-np.sin(phi), 0, np.cos(phi)]])
    r_x = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    return np.dot(r_z, np.dot(r_y, r_x))

def load_and_transform_stl(file_path):
    """Load an STL file, apply a random rotation, and return vertices and faces."""
    mesh = trimesh.load(file_path)
    rotation_matrix = random_rotation_matrix()
    rotated_vertices = np.dot(mesh.vertices, rotation_matrix.T)
    return rotated_vertices, mesh.faces

def plot_3d(vertices, faces, image_filename):
    """Create a 3D plot of the vertices and faces with a random camera angle, apply a gradient from dark grey to light grey based on z-axis, hide the internal lines, and save the plot."""
    fig = plt.figure(figsize=(7, 7), dpi=50)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=np.random.uniform(20, 40), azim=np.random.uniform(0, 360))
    
    # Normalize the z-coordinates of the vertices for gradient calculation
    z = vertices[:, 2]
    z_normalized = (z - min(z)) / (max(z) - min(z))
    colors = plt.cm.gray(z_normalized)
    
    # Find the average z position of each face for color mapping
    face_colors = colors[faces].mean(axis=1)
    
    # Create a collection for the 3D polygons with gradient coloring and no edges
    mesh = Poly3DCollection(vertices[faces], facecolors=face_colors, linewidths=0, edgecolors='none')
    ax.add_collection3d(mesh)

    # Hide the axes
    ax.set_axis_off()
    plt.tight_layout(pad=-1.5)

    # Set the viewing scale based on the mesh size
    min_val = np.min(vertices, axis=0)
    max_val = np.max(vertices, axis=0)
    max_range = max(max_val - min_val)  # Maximum range for x, y, and z
    mid_val = (max_val + min_val) * 0.5
    ax.set_xlim(mid_val[0] - max_range / 2, mid_val[0] + max_range / 2)
    ax.set_ylim(mid_val[1] - max_range / 2, mid_val[1] + max_range / 2)
    ax.set_zlim(mid_val[2] - max_range / 2, mid_val[2] + max_range / 2)
    # Save the figure with high resolution to capture more details
    plt.savefig(image_filename, dpi=50, bbox_inches='tight', pad_inches=-1.5)
    plt.close()

# Example usage
def create_image_grid(file_path, output_path, task_count, grid_size=(7, 7), image_size=(512, 512)):
    print(file_path)
    """Generate random perspective projections and arrange them in a grid."""
    vertices, faces = None, None
    vertices, faces = load_and_transform_stl(file_path)
    grid_image = Image.new('RGB', (image_size[0] * grid_size[0], image_size[1] * grid_size[1]))
    for i in range(grid_size[0] * grid_size[1]):
        image_filename = "temp_dir/temp_image_"+str(task_count)+"__"+str(i)+".png"
        plot_3d(vertices, faces, image_filename)
        with Image.open(image_filename) as img:
            img_resized = img.resize(image_size)
            row = i // grid_size[0]
            col = i % grid_size[1]
            grid_image.paste(img_resized, (col * image_size[0], row * image_size[1]))
        os.remove(image_filename)  # Delete the temporary image to free up disk space

    grid_image.save(output_path)
    vertices, faces = None, None
    gc.collect()


# Example usage
#file_path = 'C:/Users/chpre/OneDrive/Desktop/Image_guided_diffusion/Dataset/cell_count_1_2/design_BCC90_stl.stl'
#create_image_grid(file_path)

def process_directory(directory_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    tasks = []
    task_count = 0
    for filename in os.listdir(directory_path)[0:911]:
        #print(filename)
        if filename.endswith('.stl'):
            stl_path = os.path.join(directory_path, filename)
            out_img_filename = f"{os.path.splitext(os.path.basename(stl_path))[0]}.png"
            output_save_path = os.path.join(output_path, out_img_filename)
            tasks.append((stl_path, output_save_path, task_count))
            task_count = task_count+1
    
    with Pool(processes=3) as pool:
        pool.starmap(create_image_grid, tasks)
    gc.collect()

if __name__ == '__main__':
    freeze_support()  # For Windows support when using multiprocessing
    input_directory = 'C:/Users/chpre/OneDrive/Desktop/Image_guided_diffusion/Dataset/cell_count_1_2/'  # Update this path
    output_directory = 'C:/Users/chpre/OneDrive/Desktop/Image_guided_diffusion/Dataset/images/'  # Update this path
    process_directory(input_directory, output_directory)
