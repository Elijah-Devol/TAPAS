# tsdf_fusion.py

## Purpose
Builds per-trajectory TSDF (Truncated Signed Distance Function) reconstructions
of the scene from RGB-D frames, clusters the resulting point cloud into
candidate objects, and projects the clusters back into every camera view to
generate per-pixel object masks. These masks are then stored on the dataset for
downstream consumption (e.g. as the "GT" segmentation supervision used by
dense-correspondence training).

## What it does
For each trajectory in the dataset:
1. Loads the fusion-camera views (typically the wrist cam, possibly subsampled
   for pose/content) and stacks RGB/depth/intrinsics/extrinsics.
2. Calls `tsdf.fusion.fuse(...)` with the env-specific `coordinate_box` and
   `gripper_dist` bounds (from `tapas_gmm.tsdf.filter`) to integrate into a
   TSDF volume on GPU.
3. Extracts a point cloud and a triangle mesh from the TSDF; writes mesh and
   pointcloud `.ply` files next to the dataset (for inspection only).
4. Filters the supporting table plane out of mesh+pointcloud with
   `filter_plane_from_mesh_and_pointcloud`.
5. Runs DBSCAN-like `tapas_gmm.tsdf.cluster.Cluster.fit` (`eps=0.03`,
   `min_samples=5000`) to label point-cloud points as belonging to objects.
6. Loads the full scene (all cameras, no subsampling) and calls
   `fusion.build_masks_via_mesh(...)` to project the labelled mesh into every
   camera view, producing a per-pixel mask.
7. Splits the masks back per-camera per-trajectory and writes them to disk via
   `scene_data.add_tsdf_masks(...)`.

## Key components
- `FusionConfig`: which cameras to use for fusion vs. mask generation, and how
  to subsample.
- `Config`: data-naming + env identifier + `FusionConfig`.
- `main`: orchestrates per-trajectory fusion, clustering, and mask projection;
  wrapped in `@torch.no_grad()` and uses CUDA-context helpers from
  `utils.cuda` for the optional pycuda context.

## Notable details
- Reads coordinate-box/gripper-distance defaults per environment from
  `tapas_gmm.tsdf.filter.coordinate_boxes` / `gripper_dists`.
- The mesh/pointcloud `.ply` files are intentionally not added to the dataset
  itself - they are scratch artefacts for verification.
