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
written to a temporary directory; existing datasets were not overwritten.

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

The latest checkout and the previous Toolkit copy used by RACE-6D were both run
on the same 3,999-row prediction CSV, using the same LMO test data, BOP19 targets, C++
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
  shared. Importing from outside the checkout selects this checkout and yields
  dataset root `~/Downloads` and renderer path `~/Downloads/bop_renderer/build`.
- A regular wheel was built and installed into an isolated target directory.
  Its module entry point successfully exported one actual LMO image/eight
  annotations from outside the checkout.
- The existing Python 3.10 C++ renderer generated a finite 640x480 depth image
  with 2,770 nonzero pixels using the default renderer path, without an override.

## Scope of verification

The full LMO evaluation split was converted and compared. PBR checks used small
samples across the six other datasets; full training-set conversion and model
training were not part of this stage (see the
[full-data validation](#full-data-validation-2026-10-12) below). The exporter
requires the original raw split, even when a previously converted COCO dataset
is present.

The converter supports classic BOP/pinhole data, with explicit errors for
unsupported camera/sensor layouts or missing source data. The 2026-09-09 scope
did not generate keypoints. The Toolkit model-preparation extension below now
provides that separate stage without a RACE6D notebook dependency.

## Rotation preparation extension (2026-10-11)

This section records the new optional rotation policy; it does not replace the
2026-09-09 conversion/evaluation results above.

The focused conversion, rotation and upstream I/O suite passed **37 tests**:

```bash
.venv/bin/python -m pytest -q bop_toolkit_lib/tests/test_race6d_coco.py \
  bop_toolkit_lib/tests/test_race6d_rotations.py bop_toolkit_lib/tests/test_inout.py
```

Coverage includes unchanged valid rotations (including 180-degree rotations),
small-drift SVD projection, idempotence, rejection of reflections/singular or
grossly invalid matrices, malformed/non-finite inputs, read-only audits, source
hashes, existing-report protection, deterministic serial/parallel repair logs,
and failure cleanup. CLI help, Python compilation and whitespace checks passed.

Actual-data checks, without modifying any source annotations:

- Current LM-O test GT: all 1,517 raw instances passed the `1e-6` rotation
  tolerance. Three selected images/24 exported annotations were converted under
  `preserve`, `check` and `normalize` into separate temporary directories. COCO
  content was identical apart from creation timestamps, and the source SHA256
  was unchanged. No rotation needed repair in this already-normalized source.
- Earlier annotation copies normalized by a previous float32 6D pipeline were
  checked and projected **in memory only**, to exercise realistic drift rather
  than only synthetic data:

| Backup sample | Raw instances | Failed tolerance | Accepted bounded projections | Maximum matrix-entry change |
| --- | ---: | ---: | ---: | ---: |
| LM-O test, scene 2 | 1,517 | 1,517 | 1,517 | 0.00458306 |
| HB val_primesense, scene 3 | 1,360 | 1,280 | 1,280 | 0.00000123270 |
| TUD-L train_real, scene 0 | 13,426 | 13,426 | 13,426 | 0.00661878 |

No projection exceeded the former `0.01` entry-change bound in these samples (the full TUD-L `train_real` split reaches `0.0118`, hence the `0.02` default);
all source hashes remained unchanged. Passing SO(3) checks confirms matrix
validity, not that a pose label is semantically correct. The new float64 SVD
method is not claimed to reproduce historical float32 6D normalization exactly.

All 20 dataset/split combinations documented in the new recipe tables/examples
were accepted by `dataset_params.get_split_params`. This is a path/parameter
check, **not** full conversion of those datasets. Full normalized training
exports, keypoint regeneration, training and fresh BOP evaluation were not run.

## Training mesh/keypoint preparation extension (2026-10-11)

The Toolkit now ports RACE-6D's former P and 32-keypoint computations into
`race6d_models.py`, `race6d_keypoints.py` and `scripts/prepare_race6d_models.py`.
This is a new, guarded asset preparation path, not replacement of previously
validated datasets/caches/checkpoints. (The bundle at this stage also contained
shifted mesh copies; it now holds only P and the original-frame keypoint cache.)

The combined rotation, conversion, asset and upstream-I/O suite passed **51
tests**. Checks include all-symmetry constraint solving, identity/reflection
handling, residual rejection, model-frame metadata transformation, projection
equivalence `R(v-P)+(t+RP)=Rv+t`, inverse P correction, PLY property/source
preservation, strict 32-point/12-edge geometry, duplicate/invalid edge rejection,
publication only on success, existing-output and repeated-shift rejection.
COCO tests also verify different per-image K values are preserved and invalid
pinhole focal lengths/bottom rows are rejected. Off-image principal points
remain allowed; no camera unification is performed.

Actual-data checks ran in temporary output directories, not over source data:

- **LM-O asset bundle:** all 8 selected models prepared with models and
  models_eval copies, P, keypoint cache, integration contract and manifest.
  All 18 source mesh/metadata hashes were unchanged. The default threshold
  selected P=0 for these LM-O objects. Each cache entry has 32 distinct points
  and 12 valid edges.
- **Notebook port parity:** execute only the legacy notebook's function
  definitions in an isolated namespace, never its dataset/save cells. For
  the 8 LM-O models, the maximum new-versus-notebook coordinate difference was
  **0 mm**, and all edge index arrays were identical. This covers those local
  LM-O inputs/presets, not arbitrary meshes or historical cache versions.
- **TUD-L / IC-BIN:** read-only in-memory cache generation checked all 3 / 2
  local models respectively; no dataset/cache was rewritten.
- **COCO smoke check:** 3 actual LM-O test images, 24 annotations, depth paths,
  `--rotation_mode normalize`. No rotation required repair. Each annotation's
  K exactly matched its original image's scene_camera entry.

Optional geometry dependencies are installed locally and declared in the
`scripts` extra. Compilation, CLI help and diff whitespace checks passed.
No original model, scene GT, camera, image, depth or existing keypoint cache
was modified. Full asset generation for the remaining datasets, full new
training-set conversion, missing-mask rendering, model training and BOP score
evaluation were not run in this extension. Raw BOP inputs must include the
documented prerequisites; missing/private GT cannot be generated by this tool.


### YCB-V / ITODD compatibility follow-up (2026-10-11)

This describes the **pre-fix** actual-data follow-up. It found gaps, so at that
point the model preparer was not a drop-in replacement for both datasets.
No original data or implementation was changed during those diagnostic tests.
The later notebook-compatible fix and fresh validation are recorded below.

- YCB-V: 42 recovered original PLYs and both metadata JSONs match the existing
  original ZIP byte-for-byte. Default preparation stops at object 16 because
  its approximate symmetry equations have a `0.273802 mm` residual, above the
  new `0.01 mm` guard. With an **explicit diagnostic-only** `0.3 mm` bound,
  all 21 objects generate; historical P yields notebook/saved-cache parity
  within `3.3e-14 mm`, with identical edge indices. This is not a default
  success or a recommendation to loosen every dataset's guard.
- ITODD: all 28 assets generate under the default guard. P matches the old
  computation; 26 keypoint caches match. Objects 12/28 differ because the
  port combines continuous and discrete symmetries while the old notebook
  prioritizes continuous symmetry for its box frame. Max same-index point
  differences are `27.988738 / 23.456976 mm`. An isolated continuous-priority
  control restores **0 mm** difference and identical edges for both. Strict
  valid geometry does not imply trained-checkpoint compatibility.
- Mesh properties and paired `v-P`, `t+R@P` geometry are preserved. Actual PBR
  instances of YCB-V 19/20 and ITODD 2 differ in projection by less than
  `0.00000569 px` after float32 PLY storage. K is untouched.
- YCB-V generated PLYs still refer to 21 texture PNGs that the preparer does
  not copy. Geometry-only checks pass, but this is not a complete textured
  rendering bundle.

The 51 tests pass again; the newly discovered dataset compatibility gaps were
outside those tests.

### Notebook-compatible generation fix (2026-10-11)

The source of truth is RACE-6D's keypoint notebook (`RACE6D/sub/test_kpt.ipynb`), not
merely a symmetry-equivalent alternative keypoint box. The port was corrected
to keep its algorithm on valid inputs:

- First continuous symmetry wins for keypoint-frame selection; otherwise use
  the legacy discrete least-squares/SVD axis choice. Continuous box evaluation
  uses only the same 90/180/270 rotations. Combining discrete flips into that
  policy was removed. This restores ITODD 12/28's original frame and ordering.
- Retain the sampled input dtype and the notebook's axis-candidate validity
  rule. The XY search, 32-point construction, split/manual-edge presets,
  temporary `(points-p)@Q` calculation and return `keypoints@Q.T+p` remain.
- The all-constraint P audit is separate. Its `0.01 mm` residual bound still
  blocks actual origin shifts. An unchanged P=0 object records its residual
  rather than failing the entire bundle. YCB-V 16 can therefore keep the
  notebook's nonzero temporary fitted center without moving the model file
  or relaxing the bound globally. Invalid structural data and invalid output
  geometry still fail instead of silently publishing a padded cache.

All three fresh default bundles passed, using original-frame sources:

| Dataset | Objects | Max point distance vs rerun notebook | Edge/frame ordering |
| --- | ---: | ---: | --- |
| LM-O | 8 | 0 mm | Identical |
| YCB-V | 21 | 2.14e-14 mm | Identical |
| ITODD | 28 | 0 mm | Identical |

YCB-V 16 itself has **0 mm** difference and the same temporary axis center
`[-0.3062975,-0.3582614,-0.2204780] mm`; model-origin P remains zero. ITODD
12/28 now match their original cache with **0 mm** difference and identical
edge indices, not merely equality after an arbitrary continuous rotation.
YCB-V's newly recomputed unrounded P differs slightly from the old saved
rounded P; max saved-cache point distance is `4.20e-9 mm`. Exact historical
reproduction still uses the P recorded with that checkpoint.

Additionally, 24 shared-input-cloud comparisons using sampling seeds 0, 17
and 2026 on representative LM-O, YCB-V and ITODD objects passed; maximum point
distance was below `3.0e-14 mm`. The notebook received the **same points** as
the new implementation for each comparison. No save/visualization cells ran.
The seven datasets' split/edge presets and unchanged construction primitives
were also compared to the notebook's source, independently of sampling.

The focused suite now passes **54 tests**, including new continuous-priority,
point-dtype and approximate-symmetry/no-shift regressions. Syntax and whitespace
checks pass. Original model/metadata/texture/cache hashes are unchanged; no
existing dataset, trained cache, checkpoint or notebook was replaced.

These checks do not establish training accuracy or external texture copying.

## Full-data validation (2026-10-12)

The seven local datasets were first compared with the official BOP archives
(CRC32 of every file of the splits and models used here); the checks below read
only verified original files.

- **COCO export**, `--rotation_mode normalize`, symlinked images: every PBR
  split (LM-O: 50,000 images, 354,291 annotations), TUD-L/YCB-V `train_real`, the BOP19 test targets of LM-O, TUD-L, IC-BIN,
  YCB-V and T-LESS, and the HB/ITODD validation splits (HB `val_primesense`:
  4,420 images, 23,103 annotations; ITODD `val`: 54 images, 123 annotations).
  No broken image links; image and annotation counts equal the source GT minus
  instances with empty masks.
- Compared with the earlier RACE-6D exports, poses, boxes, masks, visibility
  and depth scale are identical except for the rotation normalization (at most
  `3.3e-4` per matrix entry; TUD-L `train_real` carries a ~1.2% uniform scale)
  and YCB-V `train_real` scenes 60-91, where the earlier export had converted
  the poses to a single camera matrix and now keeps the original K (the earlier
  LM-O PBR export also had K rounded to within `1e-5`).
- RACE-6D's `CocoDetection` loaded the exports (RGB and RGB-D); linked images
  are pixel-identical to the earlier copied images.
- **P and keypoints** for all seven datasets: P is nonzero only for YCB-V 19/20
  and ITODD 2 and equals the values used before (within `4e-7` mm). The
  keypoint caches equal RACE-6D's previous caches (YCB-V/ITODD after moving them
  back by P), with identical edges.
