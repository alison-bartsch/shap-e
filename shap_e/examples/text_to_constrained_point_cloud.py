import os
import torch
import pymeshfix
import numpy as np
import open3d as o3d
from shap_e.diffusion.sample import sample_latents
from shap_e.util.notebooks import decode_latent_mesh
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

# setup the models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device) # reconstruction
model = load_model('text300M', device=device) # text embedding
diffusion = diffusion_from_config(load_config('diffusion')) # latent diffusion

# define the text prompt and generate
batch_size = 4
guidance_scale = 15.0

'''
Prompt ideas:
    * a pyramid
    * a cat
    * a donut
    * a car
    * a house
    * a bowl
    * a fork
    * an airplane
    * an apple
    * a teapot
    * a sculpture
    * a statue
    * pottery
    * a clay sculpture
    * modern art sculpture
    * a human sculpture
    * a vase
    * a dinosaur
'''
prompt = "a bowl" 

latent = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)
print("\nLatent shape: ", latent[0].shape)

for j in range(batch_size):
    # decode to mesh format
    t = decode_latent_mesh(xm, latent[j]).tri_mesh()
    with open('intermediate_mesh.ply', 'wb') as f:
        t.write_ply(f)
    mesh = o3d.io.read_triangle_mesh('intermediate_mesh.ply')
    o3d.visualization.draw_geometries([mesh])

    # clean the mesh
    v, f = pymeshfix.clean_from_arrays(mesh.vertices, mesh.triangles)
    fixed_mesh = o3d.geometry.TriangleMesh()
    fixed_mesh.vertices = o3d.utility.Vector3dVector(v)
    fixed_mesh.triangles = o3d.utility.Vector3iVector(f)
    # o3d.visualization.draw_geometries([fixed_mesh])

    # simplify the mesh
    fixed_mesh = fixed_mesh.simplify_quadric_decimation(2048)
    o3d.visualization.draw_geometries([fixed_mesh])

    # # extra simplification test
    # fixed_mesh = fixed_mesh.simplify_quadric_decimation(100)
    # o3d.visualization.draw_geometries([fixed_mesh])

    # get voxel from mesh
    voxel = o3d.geometry.VoxelGrid.create_from_triangle_mesh(fixed_mesh, voxel_size=0.1)
    o3d.visualization.draw_geometries([voxel])

    # get a point cloud from the voxels
    pcl = np.asarray([voxel.origin + pt.grid_index*voxel.voxel_size for pt in voxel.get_voxels()])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcl)
    o3d.visualization.draw_geometries([point_cloud])

    # check if the mesh is watertight
    watertight = fixed_mesh.is_watertight()
    print("\nWatertight: ", watertight)

    if watertight:
        # calculate the mesh volume
        v = o3d.geometry.TriangleMesh.get_volume(fixed_mesh)
        print("\nVolume: ", v)

        # scale the mesh to a specified volume
        target_volume = 5
        scale_factor = (target_volume / v) ** (1/3)
        fixed_mesh.scale(scale_factor, center=fixed_mesh.get_center())
        o3d.visualization.draw_geometries([fixed_mesh])
        print("\nVolume: ", o3d.geometry.TriangleMesh.get_volume(fixed_mesh))

    else:
        # try to fill small boundaries (assume there are some kind of holes in the mesh)
        # could maybe even do some kind of iterative filling (i.e. slowly decrease nbe parameter in fill_small_boundaries) until watertight?
        
        # check if bad_meshi.ply exists, if it does, increase i by 1
        i = 0
        while os.path.exists(f'bad_meshes/bad_mesh{i}.ply'):
            i += 1
        # save the fixed_mesh to bad_meshes file as a .ply using open3d
        o3d.io.write_triangle_mesh(f'bad_meshes/bad_mesh{i}.ply', fixed_mesh)

    # sample the mesh to a point cloud
    alternate_point_cloud = fixed_mesh.sample_points_poisson_disk(2048)
    o3d.visualization.draw_geometries([alternate_point_cloud])