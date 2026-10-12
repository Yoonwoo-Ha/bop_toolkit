# RACE-6D data preparation

This branch starts from BOP Toolkit commit
`cea62d651c7e395b2e1962b9749e4e89693c6ac4` and adds the dataset export and model
assets needed by [RACE-6D](https://github.com/Yoonwoo-Ha/RACE6D). BOP pose-error,
matching, scoring and renderer implementations keep their upstream behavior.

Starting from the original BOP archives, two commands prepare everything RACE-6D
trains and evaluates on:

1. `scripts/prepare_race6d_coco.py` exports a **new** merged COCO dataset per split
   (RGB, optional depth, poses, per-image `cam_K`, masks). Images are linked to
   the original files by default. Slightly invalid rotations can be normalized in
   the exported annotations only.
2. `scripts/prepare_race6d_models.py` computes the model-frame shift P and the
   32-keypoint/12-edge cache in the original model frame, and installs both in
   the dataset folder that RACE-6D reads.

Neither command modifies the original BOP files: images, `scene_gt.json`,
`scene_camera.json`, masks and meshes stay as distributed, and BOP scoring uses
them unchanged. Each image keeps its **own `cam_K`**; train/test intrinsics are
not unified. P is applied by RACE-6D when it loads the data and reversed in its
BOP output.

## Installation

Clone the RACE-6D branch of the fork, then create an environment and install
the package:

```bash
git clone --branch race6d-data https://github.com/Yoonwoo-Ha/bop_toolkit.git \
  "$HOME/Downloads/bop_toolkit_race6d"
cd "$HOME/Downloads/bop_toolkit_race6d"
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[scripts]'
python -c 'from bop_toolkit_lib import config; print(config.__file__); print(config.datasets_path)'
```

The `scripts` extra adds the geometry dependencies of the model preparer
(Open3D, trimesh, plyfile); the COCO exporter does not need them. The printed
module must belong to this checkout; use the same environment for the scripts
and the package. An older installed Toolkit can have a different renderer
module layout.

If reusing system packages with `venv --system-site-packages`, an existing
Toolkit installation can take precedence over an editable import hook. Use
`python -m pip install --no-deps -e . --config-settings editable_mode=strict`
in that environment and check the printed path from outside the checkout too.
Strict mode points into this checkout's `build/__editable__...` directory;
reinstall after adding new package files. `--no-deps` is only appropriate when
the required dependencies are already installed.

Dataset archives and the C++ renderer are installed separately. The COCO
exporter reads existing GT and mask files, so it runs without a renderer; the
renderer is needed for VSD in BOP evaluation and for regenerating masks.

## Paths and source data

Datasets are read from `$BOP_PATH/<dataset>/`; without `BOP_PATH` the default is
`~/Downloads/<dataset>/`. `--datasets_path` overrides both for one command:

```bash
export BOP_PATH="/data/bop"   # /data/bop/lmo, not /data/bop/lmo/lmo
```

Extract the official archives with their split names, for example:

```text
$BOP_PATH/lmo/
  models/
  models_eval/
  test_targets_bop19.json
  train_pbr/000000/...
  test/000002/
    rgb/  depth/  mask/  mask_visib/
    scene_gt.json  scene_gt_info.json  scene_camera.json
```

Each converted scene needs GT poses, visibility information, camera intrinsics,
visible masks, full masks and the source images; RGB-D export also needs depth
images. The exporter supports the classic BOP datasets with pinhole cameras
(LM-O, TUD-L, IC-BIN, ITODD, YCB-V, T-LESS, HB). Multi-sensor and fisheye
datasets need a separate conversion.

The official test GT of **HB and ITODD** is not public. Their annotated
official validation splits (`hb/val_primesense`, `itodd/val`) are the evaluation
sets during training; they are exported to the same folder name as the test
evaluation sets of the other datasets (`<dataset>2coco_test_eval_amodal`), which
the RACE-6D configs read. Test scores come from the BOP server.

RACE-6D's YAML paths are configured separately; `BOP_PATH` does not rewrite them.

## Prepare all RACE-6D data

Run from this checkout with its environment active. Each command writes one
new COCO directory under `$BOP_PATH/<dataset>/` and refuses an existing one.

```bash
COCO="python scripts/prepare_race6d_coco.py --rotation_mode normalize --num_workers 8"

# Evaluation: BOP19 test targets (LM-O, TUD-L, IC-BIN, YCB-V, T-LESS) or the
# official validation split (HB, ITODD)
$COCO --dataset lmo   --split test --include_depth
$COCO --dataset tudl  --split test --include_depth
$COCO --dataset icbin --split test --include_depth
$COCO --dataset ycbv  --split test --include_depth
$COCO --dataset tless --split test --split_type primesense --include_depth
$COCO --dataset hb    --split val  --include_depth --output_dir "$BOP_PATH/hb/hb2coco_test_eval_amodal"
$COCO --dataset itodd --split val  --output_dir "$BOP_PATH/itodd/itodd2coco_test_eval_amodal"

# Training: PBR for all datasets, plus the real training images of TUD-L and YCB-V
for ds in lmo tudl icbin itodd ycbv tless hb; do
  $COCO --dataset $ds --split train --split_type pbr --include_depth
done
$COCO --dataset tudl --split train --split_type real --include_depth
$COCO --dataset ycbv --split train --split_type real --include_depth

# P and keypoints, installed in each dataset folder
for ds in lmo tudl icbin itodd ycbv tless hb; do
  python scripts/prepare_race6d_models.py --dataset $ds \
    --models_path "$BOP_PATH/$ds/models" --models_eval_path "$BOP_PATH/$ds/models_eval" \
    --output_dir "$BOP_PATH/$ds/race6d_models" --install_to "$BOP_PATH/$ds"
done
```

Drop `--include_depth` for RGB-only training. ITODD real depth is stored in
fractional millimeters and cannot be written losslessly as uint16 PNG, so its
`val` (and `test`) split is exported without depth; the exporter stops with an
error instead of rounding it.

## COCO export

### Splits and outputs

| Dataset | Arguments | Raw split | Default output |
| --- | --- | --- | --- |
| LM-O | `--dataset lmo --split test` | `lmo/test/` | `lmo2coco_test_eval_amodal/` |
| TUD-L | `--dataset tudl --split test` | `tudl/test/` | `tudl2coco_test_eval_amodal/` |
| IC-BIN | `--dataset icbin --split test` | `icbin/test/` | `icbin2coco_test_eval_amodal/` |
| YCB-V | `--dataset ycbv --split test` | `ycbv/test/` | `ycbv2coco_test_eval_amodal/` |
| T-LESS | `--dataset tless --split test --split_type primesense` | `tless/test_primesense/` | `tless2coco_test_eval_amodal/` |
| HB | `--dataset hb --split val --output_dir $BOP_PATH/hb/hb2coco_test_eval_amodal` | `hb/val_primesense/` | (default `hb2coco_val_primesense_amodal/`) |
| ITODD | `--dataset itodd --split val --output_dir $BOP_PATH/itodd/itodd2coco_test_eval_amodal` | `itodd/val/` | (default `itodd2coco_val_amodal/`) |
| PBR (all) | `--dataset <ds> --split train --split_type pbr` | `<ds>/train_pbr/` | `<ds>2coco_train_pbr_amodal/` |
| TUD-L real | `--dataset tudl --split train --split_type real` | `tudl/train_real/` | `tudl2coco_train_real_amodal/` |
| YCB-V real | `--dataset ycbv --split train --split_type real` | `ycbv/train_real/` | `ycbv2coco_train_real_amodal/` |
| T-LESS real | `--dataset tless --split train --split_type primesense` | `tless/train_primesense/` | `tless2coco_train_primesense_amodal/` |

- **test** selects the images listed in `<dataset>/test_targets_bop19.json` and
  keeps every eligible instance in them (`--targets_filename` sets another list).
- **val** uses every annotated image; BOP validation splits have no target list.
  `--targets_filename` restricts it to a list.
- **train** always uses every annotated image present, ordered by the original
  scene/image IDs; it does not assume scenes start at `000001`.
- `--use_all_gt` makes a test export use every annotated image; the default
  output name then follows the split folder.

Evaluation outputs contain `val2017/` and `annotations/instances_val2017.json`,
training outputs `train2017/` and `annotations/instances_train2017.json`.
`--include_depth` adds the sibling `val2017_depth/` or `train2017_depth/` that
RACE-6D reads for RGB-D. An RGB-only output must be regenerated at a new path to
add depth. LM-O keeps its eight BOP object IDs. ITODD PBR images are RGB; its
real images are grayscale TIFFs and keep their extension.

Without `--split_type`, train means PBR and test/val use the Toolkit's default
sensor (Primesense for T-LESS and HB). `--split_type` selects a suffixed folder,
e.g. `real`; other folder names are not treated as aliases, and the converter
does not guess a sensor or rename raw data.

### Options

- `--image_mode symlink` (default) links each image to its original file with
  an absolute path; keep the original dataset in place. `copy` makes a
  self-contained output.
- `--scene_ids`, `--image_ids` and `--max_images` select subsets; use a new
  `--output_dir` for them, e.g.

  ```bash
  python scripts/prepare_race6d_coco.py --dataset lmo --split test --max_images 3 \
    --include_depth --output_dir "$BOP_PATH/lmo/lmo2coco_test_eval_amodal_check"
  ```

- `--num_workers` converts scenes in parallel (0 = CPU count). Scene/image order
  and IDs do not depend on it, and annotations are streamed, so a training split
  is never held in memory.
- `python scripts/prepare_race6d_coco.py --help` lists all options.

A conversion is built in a temporary sibling directory and published only when
it succeeds. `conversion_summary.json` in the output records the counts, the
selection, the rotation policy and the source `scene_gt.json` hashes.

The exporter reads GT and masks directly and writes the merged layout; per-scene
`scene_gt_coco_*.json` files (upstream [calc_gt_coco.py](scripts/calc_gt_coco.py))
are neither needed nor modified.

### Rotation checks and normalization

| `--rotation_mode` | Behavior |
| --- | --- |
| `preserve` (default) | Keep source rotations exactly, without SO(3) checks |
| `check` | Keep valid rotations; stop on an invalid exported rotation |
| `normalize` | Keep valid rotations; project slightly invalid ones to the nearest proper rotation in the new output |

A rotation is valid when its nine entries are finite and `R.T @ R ≈ I`,
`det(R) ≈ 1`, its row/column norms are 1 and `trace(R)` is in `[-1, 3]`, within
`--rotation_tolerance` (default `1e-6`). Normalization uses float64 SVD
(`U @ V.T`) on all nine entries, the nearest SO(3) matrix in Frobenius norm. It
rejects non-finite or malformed matrices, reflections and singular matrices,
and any repair that changes an entry by more than `--rotation_max_change`
(default `0.02`, an entry bound, not an angle). The official TUD-L `train_real`
labels need it: 403 of 38,288 rotations change by up to `0.0118` because they
carry a uniform scale of about 1.2% (det 1.015-1.036); SVD and Gram-Schmidt
projections agree within 0.021 deg. All other official splits stay below
`0.005`. Investigate larger errors instead of raising the bound. Valid matrices
are unchanged, so normalizing twice does not move a pose.

Conversion checks only **exported** annotations (instances with a category and
non-empty masks). In `check`/`normalize` mode, `rotation_changes.jsonl` lists
every changed rotation with its before/after values, errors and original
scene/image/GT identity. Translations, intrinsics, boxes and masks are not
adjusted, and the source GT is never rewritten.

To audit a whole raw split without exporting images (all GT instances, including
invisible ones):

```bash
python scripts/check_race6d_rotations.py \
  --input "$BOP_PATH/tudl/train_real" \
  --report /path/to/new/tudl_train_real_rotation_audit.json
```

The input is a split directory, a scene directory or one `scene_gt.json` (not
recursive, not a merged COCO file). Each scene report has the source SHA256,
counts, maximum errors and up to 20 invalid examples (`--max_examples`). The
exit status is `1` when an invalid rotation or an input error is found; the
report is still written, and an existing report is never overwritten.

### Camera checks

Every selected image's K must have nine finite values, positive `fx`/`fy` and
the pinhole bottom row `[0, 0, 1]`; it is saved unchanged. An off-image
principal point is allowed (e.g. cropped images). This is a structural check,
not proof that the calibration matches the photograph.

### Missing masks

COCO export does not render masks or infer GT. If `mask/`, `mask_visib/` or
`scene_gt_info.json` are missing, generate them first with the upstream
[mask generator](scripts/calc_gt_masks.py) and
[GT-info generator](scripts/calc_gt_info.py) (configured through their `p`
dictionaries, with a renderer and the original meshes). Use delta **5 mm for PBR
and ITODD real, 15 mm for other real splits**. They write into the raw split, so
check the output paths first.

## P and 3D keypoints

```bash
python scripts/prepare_race6d_models.py \
  --dataset lmo \
  --models_path "$BOP_PATH/lmo/models" \
  --models_eval_path "$BOP_PATH/lmo/models_eval" \
  --output_dir "$BOP_PATH/lmo/race6d_models" \
  --install_to "$BOP_PATH/lmo"
```

`--output_dir` receives a new bundle (no mesh files; an existing directory is
refused):

```text
race6d_models/
  cached_keypoints_3d_edge.json   # original BOP model frame
  p_values.json                   # P per object ID in mm, zero included
  model_contract.json             # how P is applied in training and BOP output
  model_preparation.json          # source/product hashes, P audit, settings
```

`--install_to` copies `cached_keypoints_3d_edge.json` and `p_values.json` to the
dataset folder, the `coco_path` RACE-6D reads them from. A different existing
file is kept as `<name>.before_<timestamp>`; an identical one is left alone.

### P

An object whose symmetry axis does not pass through the model origin is trained
in a model frame moved to a point P on that axis, so that its symmetries become
pure rotations (RACE-6D uses only their rotations).

- P is set for **YCB-V 19/20 and ITODD 2** only; every other object keeps
  P = 0, and the other datasets get an all-zero `p_values.json`. T-LESS 25/26
  also have an off-origin axis, but only their body is symmetric (a 14 mm
  protrusion has no counterpart) and RACE-6D trains them as asymmetric, so they
  keep P = 0.
- For these objects P solves `(I-R_sym) P = t_sym` and the continuous-axis
  constraints from the original `models_info.json`; on a single axis, the point
  nearest the sampled surface centroid is chosen. Inconsistent constraints
  (`--max_residual_mm`, default `0.01`) stop the bundle.
- `--p_object_ids` selects other objects for this computation, e.g. for a new
  dataset (`--p_object_ids` without IDs: none).
- `--p_values file.json` sets P explicitly, e.g. `{"19": [Px, Py, Pz]}` in mm;
  omitted objects get 0 and `{}` disables every shift. Use it to reproduce a
  model trained with another P: mesh versions and centroid sampling can give
  other points on the same axis.

P is a training convention, not a correction of the BOP geometry: meshes,
`models_info.json` and GT stay in the original frame. Inputs must be
original-frame meshes; meshes moved on disk by the earlier RACE-6D scripts
(`*.backup_paligned`) and already prepared bundles are refused, so P cannot be
applied twice.

### Keypoints

32 keypoints per object: the 8 corners of a box around the sampled mesh (the
minimum-area XY rectangle and the Z range) and 2 points on each of the 12 edges
at 1/3 and 2/3. For a symmetric object the box is also fitted in its symmetry
frame (first continuous axis, otherwise the discrete symmetries) and used when
its keypoints agree better under the symmetry rotations. The box is built on a
temporary `v-P` copy of `models_eval` and moved back by `+P`; the cache metadata
records `coordinate_system: original_bop_model_frame` and the P used.

- Point ordering, the upper/lower split and the per-dataset edge presets are
  ported from RACE-6D's former keypoint notebook for `lmo`, `tudl`, `icbin`,
  `itodd`, `ycbv`, `tless` and `hb`. Open3D sampling uses seed 0, so the cache
  is reproducible. `--no_axis_box` keeps the model-frame box for every object.
- Each object must give 32 distinct finite points and 12 collinear, evenly
  spaced four-point edges (cross-ratio 4/3); a failure stops the bundle instead
  of publishing a partial or zero-padded cache. A rectangular box cannot match
  a 3-, 5- or 6-fold symmetry exactly; its symmetry error is reported.
- LM-O defaults to its eight BOP object IDs, other datasets to every ID in
  `models_info.json`; `--object_ids` selects a subset.
- For a custom mesh, `--keypoint_config` overrides the split or edge rules:

  ```json
  {
    "split_config": {
      "15": {"type": "custom_line", "point1": [60, 0], "point2": [-60, 0]}
    },
    "edge_override": {}
  }
  ```

  An edge override supplies `upper_edges`, `lower_edges` and `vertical_edges`,
  each a list of four corner-index pairs (0..3 in that half).

## Connect to RACE-6D

Point the dataset entries of the RACE-6D configuration at the exports, for
example in `configs/race6d/r50vd/race6d_r50vd_lmo_rgb.yml`:

```yaml
coco_path: ~/Downloads/lmo/
category_file: ./configs/race6d/include/lmo2coco_category.yml

train_dataloader:
  dataset:
    img_folder: ~/Downloads/lmo/lmo2coco_train_pbr_amodal/train2017/
    ann_file: ~/Downloads/lmo/lmo2coco_train_pbr_amodal/annotations/instances_train2017.json
    coco_path: ~/Downloads/lmo/
    category_file: ./configs/race6d/include/lmo2coco_category.yml
    return_depth: False

val_dataloader:
  dataset:
    img_folder: ~/Downloads/lmo/lmo2coco_test_eval_amodal/val2017/
    ann_file: ~/Downloads/lmo/lmo2coco_test_eval_amodal/annotations/instances_val2017.json
    coco_path: ~/Downloads/lmo/
    category_file: ./configs/race6d/include/lmo2coco_category.yml
    return_depth: False
```

For HB and ITODD the validation entry reads `hb2coco_test_eval_amodal/` and
`itodd2coco_test_eval_amodal/`, exported from their validation splits. Keep the
configuration's includes, transforms, model and optimizer settings, launch
RACE-6D from its own checkout so the relative category files resolve, and update
every dataset entry of a mixed (PBR + real) training configuration.

From `coco_path`, RACE-6D reads the original `models/`,
`cached_keypoints_3d_edge.json` and `p_values.json`, and applies P once:

```text
training:   model points v-P, keypoints kp-P, GT pose t+R@P
inference:  network output + the image's K -> metric t and egocentric R
BOP output: t_original = t_training - R@P
```

These paired changes keep every projection unchanged, so masks need no
regeneration. A top-level `aligned_offsets` in a RACE-6D configuration
overrides `p_values.json`, e.g. for released checkpoints trained with another P;
`aligned_offsets: none` trains every object in the original frame. 2D keypoint
targets are projected from the cache and the GT pose during training, so the
COCO annotations carry no keypoints.

## Annotation contract

Ordinary COCO fields (`bbox`, `category_id`, `segmentation`, etc.) are accompanied
by:

| Field | Meaning |
| --- | --- |
| `pose` | `[tx, ty, tz, R00, R01, R02, R10, R11, R12, R20, R21, R22]`, BOP units and object-to-camera convention |
| `cam_K` | Camera intrinsic matrix of the image, row-major, nine values |
| `depth_scale` | Raw depth-to-millimeter scale from `scene_camera.json` (`1.0` when absent) |
| `visibility` | BOP `visib_fract` |
| `segmentation` | Visible object mask |
| `full_segmentation` | Full object silhouette within the image |
| `ignore` | Whether visibility is below `0.1` |

Amodal bounding boxes are computed from the full mask with inclusive pixel
extents; COCO `area` is the visible mask area. Instances with an empty visible
or full mask are skipped. BOP object IDs are kept; RACE-6D maps them to
contiguous labels with its category YAML. Merged images keep `scene_id`,
`bop_im_id` and `original_filename` for BOP evaluation. Older exports may call
the full-mask field `full_masks`; the RACE-6D loader accepts both.

## BOP evaluation

The default C++ renderer path is `~/Downloads/bop_renderer/build`;
`BOP_RENDERER_PATH` overrides it. The native module must match this
environment's Python version.

`eval_bop19_pose.py` computes the VSD/MSSD/MSPD average recalls and their AR.
Score with the original GT and models, and keep predictions, targets, models,
renderer and options identical when comparing versions:

```bash
python scripts/eval_bop19_pose.py \
  --renderer_type cpp --num_workers 2 \
  --results_path /path/to/predictions \
  --result_filenames method_lmo-test.csv \
  --eval_path /path/to/new/evaluation-output
```

RACE-6D's per-epoch BOP evaluation uses this checkout when its
`bop_eval.toolkit_path` points here. HB and ITODD test results are scored only
by the BOP server.

## Verification

Regression tests:

```bash
python -m pytest -q bop_toolkit_lib/tests/test_race6d_coco.py \
  bop_toolkit_lib/tests/test_race6d_rotations.py \
  bop_toolkit_lib/tests/test_race6d_models.py bop_toolkit_lib/tests/test_inout.py
```

Checks on the official data of the seven datasets (2026-10):

- The dataset files used were compared with the official archives (CRC32 of
  every file); the exports read only verified original files.
- Complete exports of every PBR split, TUD-L/YCB-V `train_real`, the BOP19 test
  targets of LM-O, TUD-L, IC-BIN, YCB-V and T-LESS, and the HB/ITODD validation
  splits, with `--rotation_mode normalize`: no broken image links, and
  image/annotation counts equal to the source GT minus empty-mask instances.
  Compared with the earlier RACE-6D exports, poses, boxes, masks, visibility and
  depth scale are identical except for the rotation normalization (at most
  `3.3e-4` per entry) and YCB-V `train_real` scenes 60-91, where the earlier
  export had converted the poses to a single camera matrix; these keep their
  original K now.
- The exports load in RACE-6D's `CocoDetection`, RGB and RGB-D, and the linked
  images are pixel-identical to the earlier copied images.
- The keypoint caches of all seven datasets equal RACE-6D's previous caches (the
  YCB-V/ITODD ones after moving them back by P), and the computed P equals the
  values the YCB-V/ITODD configurations used before (within `4e-7` mm).
- BOP19 scores of an LM-O prediction file are unchanged with this checkout.

See [validation details](docs/RACE6D_VALIDATION.md) for the earlier LM-O
annotation comparison, package-install checks and the keypoint-notebook parity
tests.
