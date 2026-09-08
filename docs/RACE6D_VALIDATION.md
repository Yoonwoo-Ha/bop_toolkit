# RACE-6D port validation

Validated on 2026-09-09 with Python 3.10. The working branch is `race6d-data`,
based on upstream `cea62d651c7e395b2e1962b9749e4e89693c6ac4`.

## Automated tests

```bash
.venv/bin/python -m pytest \
  bop_toolkit_lib/tests/test_race6d_coco.py \
  bop_toolkit_lib/tests/test_inout.py -q
```

Result: **18 passed**. This includes the 13 new regression cases for upstream
and legacy annotation calls, masks, pose/camera fields, target filtering,
deterministic serial/parallel merging, image/depth handling, and failure cleanup.
Five upstream I/O tests also passed. Python compilation and `git diff --check`
passed.

An additional independent synthetic check covered repeated original image IDs
across scenes, nontrivial rotation and translation, image-only evaluation-target
selection, LMO category filtering, unchanged source files, existing-output
protection, and cleanup after a worker encounters a missing mask.

## Actual dataset conversion

Source files were read from `~/Downloads/<dataset>/`. All test outputs were
written under `/tmp/race6d-bop-validation-1ckzd96q/`; existing datasets were not
overwritten.

| Dataset/split | Scenes | Images | Exported annotations |
| --- | ---: | ---: | ---: |
| LMO test, official BOP19 target images | 1 | 200 | 1,514 |
| YCB-V train_pbr sample | 2 | 2 | 33 |
| T-LESS train_pbr sample | 2 | 2 | 31 |
| TUD-L train_pbr sample | 2 | 2 | 6 |
| HB train_pbr sample | 2 | 2 | 36 |
| IC-BIN train_pbr sample | 2 | 2 | 40 |
| ITODD train_pbr sample | 2 | 2 | 31 |

The LMO export was compared against the existing
`lmo/lmo2coco_test_eval_amodal/annotations/instances_val2017.json`:

- All 200 source-image identities and 1,514 annotations matched.
- `category_id`, `bbox`, `pose`, `cam_K`, `depth_scale`, `visibility`, `area`,
  `ignore`, `iscrowd`, width, and height matched exactly.
- Decoded visible and full masks matched exactly. The earlier full-mask key
  `full_masks` was compared with the new `full_segmentation` key.

Each PBR sample selected image 0 from scenes 0 and 1, with two workers and depth
export enabled. All exported annotations were checked against raw scene GT,
camera metadata, and mask PNGs. Poses and camera fields were unchanged; amodal
boxes matched the inclusive full-mask extents; mask pixels and depth pixels
matched exactly. Actual JPEG image encoding and `.jpg` extensions were retained.

The actual RACE-6D `CocoDetection` class then loaded all seven exports with
visible/full masks and RGB-D enabled. Selected samples returned finite
four-channel tensors, 12-element poses, 9-element intrinsics, and the expected
depth scales (`1.0` for LMO test and `0.1` for PBR).

## Evaluation regression

The latest checkout and the prior EdgePose Toolkit copy were both run on the
same 3,999-row prediction CSV, using the same LMO test data, BOP19 targets, C++
renderer, and two workers. Both runs completed successfully.

```text
Prediction: RACE6D/bop_results/race6dr50vdrgb_lmo-test.csv
SHA-256: 72bf40a134af090de40c494c7bb7285be0812d94b6a0cd05184b1ef4471c3740
```

| Score | Previous Toolkit | Latest port |
| --- | ---: | ---: |
| VSD average recall | 0.5153148788927335 | 0.5153148788927335 |
| MSSD average recall | 0.6604152249134948 | 0.6604152249134948 |
| MSPD average recall | 0.8321799307958477 | 0.8321799307958477 |
| Overall AR | 0.6693033448673585 | 0.6693033448673585 |

The two `scores_bop19.json` dictionaries were exactly equal. Evaluation logs
are `eval_latest.log` and `eval_legacy.log` under the validation output directory.
Metric, matching, scoring, rendering, and BOP evaluation scripts were checked
against upstream and have no port changes.

## Installation and C++ renderer

- Editable installation metadata identifies this checkout as `bop_toolkit_lib`,
  version `0.1.0`.
- The prepared `.venv` uses strict editable mode because system packages are
  shared. Importing from `/tmp` selects this checkout and yields dataset root
  `~/Downloads` and renderer path `~/Downloads/bop_renderer/build`.
- A regular wheel was built and installed into an isolated target directory.
  Its module entry point successfully exported one actual LMO image/eight
  annotations from outside the checkout.
- The existing Python 3.10 C++ renderer generated a finite 640x480 depth image
  with 2,770 nonzero pixels using the default renderer path, without an override.

## Scope of verification

The full LMO evaluation split was converted and compared. PBR checks used small
samples across the six locally available datasets; full training-set conversion
and model training were not run. `~/Downloads/lmo/train_pbr` is absent on this
machine, so an actual LMO PBR export could not be run. The exporter requires the
original raw split, even when a previously converted COCO dataset is present.

The converter supports classic BOP/pinhole data, with explicit errors for
unsupported camera/sensor layouts or missing source data. Separate keypoint
cache generation remains a RACE-6D prerequisite described in the preparation
guide.
