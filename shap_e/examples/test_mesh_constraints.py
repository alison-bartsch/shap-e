import open3d as o3d
import numpy as np
import pymeshfix

# Load a mesh
mesh = o3d.io.read_triangle_mesh('example_mesh_1.ply')

# Compute the vertex normals
mesh.compute_vertex_normals()

# visualize the mesh
o3d.visualization.draw_geometries([mesh])

# try to fix mesh
# fixed_mesh = pymeshfix.MeshFix(mesh.vertices, mesh.triangles)
# fixed_mesh.repair()
v, f = pymeshfix.clean_from_arrays(mesh.vertices, mesh.triangles)
print("\nTriangles shape: ", f.shape)
# go from v, f to open3d mesh
fixed_mesh = o3d.geometry.TriangleMesh()
fixed_mesh.vertices = o3d.utility.Vector3dVector(v)
fixed_mesh.triangles = o3d.utility.Vector3iVector(f)
o3d.visualization.draw_geometries([fixed_mesh])

# simplify the fixed mesh
fixed_mesh = fixed_mesh.simplify_quadric_decimation(2048)
o3d.visualization.draw_geometries([fixed_mesh])

# check if the mesh is watertight
watertight = fixed_mesh.is_watertight()
print("\nWatertight: ", watertight)

if watertight:
    # calculate the mesh volume
    v = o3d.geometry.TriangleMesh.get_volume(fixed_mesh)
    print("\nVolume: ", v)

# scale the fixed mesh to be a specified volume
target_volume = 5
scale_factor = (target_volume / v) ** (1/3)
fixed_mesh.scale(scale_factor, center=fixed_mesh.get_center())
o3d.visualization.draw_geometries([fixed_mesh])
print("\nVolume: ", o3d.geometry.TriangleMesh.get_volume(fixed_mesh))

# uniformly sample point cloud
point_cloud = fixed_mesh.sample_points_uniformly(2048)
o3d.visualization.draw_geometries([point_cloud])

# Poisson disk sample point cloud
alternate_point_cloud = fixed_mesh.sample_points_poisson_disk(2048)
o3d.visualization.draw_geometries([alternate_point_cloud])