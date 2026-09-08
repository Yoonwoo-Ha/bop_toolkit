# RACE-6D data preparation

This branch starts from BOP Toolkit commit
`cea62d651c7e395b2e1962b9749e4e89693c6ac4` and adds the annotation fields and
dataset export needed by RACE-6D. BOP pose-error, matching, scoring, and renderer
implementations retain their upstream behavior.

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
python -m pip install -e .
python -c 'from bop_toolkit_lib import config; print(config.__file__); print(config.datasets_path)'
```

The printed module must belong to this checkout. Use the same environment for
the scripts and their imported package. An older installed Toolkit can have a
different renderer module layout.

If reusing system packages with `venv --system-site-packages`, an existing
Toolkit installation can take precedence over an editable import hook. Use
`python -m pip install --no-deps -e . --config-settings editable_mode=strict`
in that environment and check the printed path from outside the checkout too.
Strict mode points into this checkout's `build/__editable__...` directory;
reinstall after adding new package files. `--no-deps` is only appropriate when
the required dependencies are already installed.

## Paths and source data

By default, datasets are in `~/Downloads/<dataset>/`, for example:

```text
~/Downloads/lmo/
  models/
  models_eval/
  test_targets_bop19.json
  train_pbr/000000/...
  test/000002/
    rgb/
    depth/
    mask/
    mask_visib/
    scene_gt.json
    scene_gt_info.json
    scene_camera.json
```

Each converted scene needs GT poses, visibility information, camera intrinsics,
visible masks, full masks, and source images. RGB-D export also needs depth
images. COCO conversion does not render missing masks or infer missing GT.
Some official test sets do not distribute pose GT; use an available annotated
split for those datasets.
The exporter supports classic BOP datasets with pinhole cameras. Multi-sensor
and fisheye datasets require a separate conversion.

`BOP_PATH` or `--datasets_path` overrides the dataset parent directory:

```bash
export BOP_PATH="/data/bop"
```

This means `/data/bop/lmo`, not `/data/bop/lmo/lmo`. RACE-6D's YAML paths are
configured separately; the environment variable does not rewrite them.

## Convert and merge

The dedicated command produces a complete COCO dataset, including merged
annotations and image paths. The original `scripts/calc_gt_coco.py` retains
the upstream BOP detection/segmentation interface.

LMO evaluation (only images in `test_targets_bop19.json`):

```bash
python scripts/prepare_race6d_coco.py \
  --dataset lmo --split test --bbox_type amodal --num_workers 4
```

Output:

```text
~/Downloads/lmo/lmo2coco_test_eval_amodal/
  val2017/
  annotations/instances_val2017.json
```

LMO PBR training (requires the original `lmo/train_pbr` data):

```bash
python scripts/prepare_race6d_coco.py \
  --dataset lmo --split train --split_type pbr \
  --bbox_type amodal --num_workers 4
```

Output is `~/Downloads/lmo/lmo2coco_train_pbr_amodal/`, containing `train2017/`
and `annotations/instances_train2017.json`. Training uses all GT images and
does not require an evaluation-target file.

For YCB-V real data, use `--dataset ycbv --split train --split_type real`.
If a test split is stored as `test_real`, explicitly pass `--split_type real`.
Toolkit dataset parameters choose the standard sensor split for datasets such
as T-LESS and HB. `--use_all_gt` includes all annotated test images instead of
selecting evaluation targets.

To inspect a small subset without replacing an existing dataset:

```bash
python scripts/prepare_race6d_coco.py \
  --dataset lmo --split test --max_images 3 \
  --output_dir "$HOME/Downloads/lmo/lmo2coco_test_eval_amodal_check" \
  --image_mode symlink --include_depth
```

Use `--scene_ids` and `--image_ids` for explicit subsets. Use `--image_mode copy`
for an output whose images do not depend on the source files remaining in place.
Image extensions match their actual encoding. Depth export creates the sibling
`train2017_depth/` or `val2017_depth/` directory expected by RACE-6D.

Existing output directories are rejected. A conversion is built in a temporary
sibling directory and published only after it succeeds. Scene/image ordering and
identifiers are deterministic across worker counts; annotations are streamed so
the entire training dataset is not held in memory.

## Annotation contract

Ordinary COCO fields (`bbox`, `category_id`, `segmentation`, etc.) are accompanied
by:

| Field | Meaning |
| --- | --- |
| `pose` | `[tx, ty, tz, R00, R01, R02, R10, R11, R12, R20, R21, R22]`, retaining BOP units and object-to-camera convention |
| `cam_K` | Camera intrinsic matrix in row-major order, nine values |
| `depth_scale` | Raw depth-to-millimeter scale from `scene_camera.json`; BOP's default is `1.0` when absent |
| `visibility` | BOP `visib_fract` |
| `segmentation` | Visible object mask |
| `full_segmentation` | Full object silhouette within the image |
| `ignore` | Whether visibility is below `0.1` |

Amodal bounding boxes are computed from the full mask, using inclusive pixel
extents, as in the original RACE-6D converter. COCO `area` describes the visible
mask. Zero-area visible/full instances are skipped. BOP object IDs are preserved;
RACE-6D handles contiguous class remapping using its category YAML.

Merged images retain `scene_id` and `original_filename` for BOP evaluation.
Older exported datasets may call the full-mask field `full_masks`; the RACE-6D
loader accepts either spelling. The standard annotation-builder calls used by
BlenderProc remain supported, as do the earlier RACE-6D positional call styles.

## Connect to RACE-6D

Set `coco_path` to `~/Downloads/lmo/` and the train/validation `img_folder` and
`ann_file` values to the output paths above. Update dataset-level `coco_path`
entries too when using another root.

RACE-6D also requires `models/` and `cached_keypoints_3d_edge.json` under
`coco_path`. The COCO converter does not generate the 32 object keypoints.
The existing generator is in RACE-6D's `sub/test_kpt.ipynb`, which reads
`models_eval/` and saves the separate keypoint/edge cache. Enable its save call
when generating a new cache; the model does not create a missing cache
automatically. Keep an existing validated cache when reproducing a checkpoint.
Per-instance 2D keypoints are projected from the cache and GT pose during
training, so they do not need to be embedded in the COCO annotation.

## C++ rendering and BOP evaluation

The default C++ renderer path is `~/Downloads/bop_renderer/build`.
`BOP_RENDERER_PATH` can override it. The native module must match the Python
version used by this environment (the local existing build uses Python 3.10).

Use `eval_bop19_pose.py` for the VSD/MSSD/MSPD average recalls and their aggregate
AR. Keep the same predictions, targets, models, renderer, and options when
comparing versions. `eval_bop24_pose.py` evaluates a different detection task.

```bash
python scripts/eval_bop19_pose.py \
  --renderer_type cpp --num_workers 2 \
  --results_path /path/to/predictions \
  --result_filenames method_lmo-test.csv \
  --eval_path /path/to/new/evaluation-output
```

EdgePose's `bop_eval.toolkit_path` must explicitly point to this checkout to use
it; its existing `tools/bop_toolkit` copy does not update automatically.

## Verification

See [validation results](docs/RACE6D_VALIDATION.md) for the actual-data checks,
LMO annotation comparison, package-install checks, and BOP score regression.
