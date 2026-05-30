# viz/ - Visualisation helpers

Matplotlib / OpenCV plotting helpers used by the scripts in
`tapas_gmm/` (especially the visualisation entry points and the
notebooks).

## Top-level files

### `action_distribution.py`
Histograms of action components - 2D histos over translation/rotation/gripper
and a per-dimension `make_all` plot used by `inspect_dataset.py`.

### `activation_map.py`
Per-channel softmax activation maps for the keypoint encoder, rendered as
seaborn jointplots with a 2D density estimate of the softmax mass and the
keypoint position overlaid.

### `correspondence_plotter.py`
Dense-correspondence training diagnostics. Renders pairs of images with
similarity heatmaps and matched/non-matched pixel markers. Used by
`DenseCorrespondenceDataset` debug plotting and the live heatmap viz.
Exports: `plot_heatmaps`, `debug_plots`, `cross_debug_plot`,
`plot_correspondences`, `plot_correspondences_direct`.

### `function.py`
Sanity utility: `show_function(f, xmin, xmax)` plots a 1D function over a
range. Used during development.

### `gmm.py` (72KB)
Plotting machinery for the TP-GMM. Plots per-component
ellipsoids/contours, per-frame trajectories, time-conditioned predictions,
TOPP-scaled trajectories (`plot_traj_topp`), per-component responsibility
masses, Riemannian-manifold projections, and GMR conditioning slices. The
GMM notebooks rely heavily on this module.

### `image_pair.py`, `image_series.py`, `image_single.py`
Small helpers to display 1, N (grid), or 2 (side-by-side) images with
optional keypoint overlays and (for `image_single`) optional embedding
overlays. Many decorated with `@mp_wrapper` so they show up in a separate
process and don't block the training loop.

### `keypoint_selector.py`
OpenCV-based interactive widget. Used by the manual reference-keypoint
selection workflow (`KeypointsPredictor` with
`ReferenceSelectionTypes.MANUAL`): click on the input image to select
reference pixels, the chosen descriptors and 3D world coordinates are
returned to the encoder.

### `live_heatmap_visualization.py`
Interactive OpenCV viewer used by `visualize_dense_correspondence.py`.
Hover the mouse over one image to see the similarity heatmap (and
best-match cross-hair) in another image, for each loaded encoder.
`HeatmapVisualizationConfig` controls the displayed channels, the
distance metric, etc.

### `live_keypoint.py`
Real-time keypoint overlay used during evaluation. Subscribes to the
policy's per-step `info` dict (which carries keypoint positions, heatmaps,
particle filter state, etc.) and renders them on top of the current
camera observation. The `LiveKeypoints.setup_from_conf(config, policy)`
classmethod selects the rendering mode based on the encoder's prior type.

### `operations.py`
Image-tensor housekeeping:
- `channel_front2back(x)` / `channel_back2front(x)` and their batched
  variants - permute `(C, H, W)` <-> `(H, W, C)`.
- `np_channel_front2back` / `np_channel_back2front` - numpy equivalents.
- `rgb2gray(x)` - luminance.
- `int_to_float_range(x)` - `[0, 255]` -> `[0, 1]`.
- `scale(x, out_range=(0,1))` - min-max rescale.
- `get_image_tensor_mean`, `get_image_tensor_std` - dataset statistics
  (used by encoder image-normalisation init).
- `uv_to_flattened_pixel_locations`, `flattened_pixel_locations_to_u_v` -
  conversions between 2D pixel positions and the flat indexes used by
  `correspondence_finder` and the dense-correspondence loss.

### `particle_filter.py`
Live visualisation of the particle filter inside `KeypointsPredictor`.
- `ParticleFilterViz`: a matplotlib animation that renders particle
  positions, weights, and the resulting posterior over multiple cameras
  per keypoint. Runs in a thread.
- `InitDbgViz`: similar widget shown during particle initialisation
  (so the user can validate the initial likelihood map).

### `point_cloud.py`
Helpers to plot two point clouds with their mutual subset highlighted
(`@mp_wrapper`'d so it runs in a child process).

### `quaternion.py`
Per-axis line plots and 3D scatter of quaternion components
(`plot_quat_time_based`, `plot_quat_imaginary_3d`,
`plot_quat_components`). Useful for sanity-checking that
`ensure_quaternion_continuity` did its job before fitting the TP-GMM.

### `surface.py`
Three-dimensional scatter / surface plots over depth maps:
- `depth_map_with_points_overlay_uv_list`: 3D mesh of the depth map with
  keypoint UV positions overlaid.
- `scatter3d`: convenience scatter wrapper.

### `threed.py`
Generic 3D-plot helpers used by the demos visualisation and the
keypoint-encoder plots:
- `plot_coordindate_frame`: renders a coordinate axis triad at a given
  pose or transformation matrix.
- 3D arrow / line classes (`Arrow3D`, `Line3D`) that play well with
  matplotlib 3D axes.
- Convenience accumulator for trajectories shown next to keypoint
  predictions.

### `utils.py`
`set_ax_border(ax, color, width)` - tiny helper to give a matplotlib axis a
coloured border (used when grids of subplots want to highlight one
row/column).
