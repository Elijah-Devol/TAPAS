# tsdf/ - TSDF fusion & post-processing

Module that supports `tsdf_fusion.py`: it integrates RGB-D frames into a
truncated signed-distance volume, extracts a mesh, clusters the mesh into
objects, and re-projects the clusters into every camera view as
per-pixel masks.

## `fusion.py` (28KB)
Adapted from Andy Zeng's TSDF-Fusion. Contains:
- The TSDF volume class (CUDA-accelerated via pycuda when available,
  numba-jitted on CPU otherwise).
- `fuse(rgb, depth, intrinsics, extrinsics, coordinate_box,
  gripper_dist)`: integrate a batch of RGB-D frames into a TSDF volume.
  `gripper_dist` excludes voxels close to the gripper position to keep
  the fingers from being reconstructed.
- `write_mesh(tsdf_vol, mesh_path)`: marching cubes via `skimage.measure`,
  saved as `.ply`. Returns vertices/faces.
- `write_pc(tsdf_vol, path)`: dump the TSDF point cloud as `.ply`.
- `build_masks_via_mesh(vertices, faces, pc_labels, intrinsics,
  extrinsics, H, W)`: rasterise the labelled mesh into per-pixel labels
  for every camera view (uses `pyrender` off-screen).
- `write_mask` helper.

## `filter.py`
- `coordinate_boxes`: per-environment workspace bounding box (the volume
  to fuse over). One entry per `Environment` enum value.
- `gripper_dists`: per-environment radius around the EE to exclude from
  fusion.
- `filter_plane_from_mesh_and_pointcloud(point_cloud, faces)`: fits a
  dominant plane to the point cloud via `open3d.geometry.PointCloud` and
  removes all vertices belonging to that plane (so the table doesn't end
  up as an "object"). Returns the filtered vertices and faces.

## `cluster.py`
Tries to import GPU-accelerated DBSCAN (`cuml.cluster.DBSCAN`) using paths
defined in the user's `conf/_machine.py`. Falls back to
`sklearn.cluster.DBSCAN` if cuml is not installed. Exports a single
`Cluster` symbol that `tsdf_fusion.py` instantiates with `eps=0.03,
min_samples=5000`.

## `__init__.py`
Empty package marker.
