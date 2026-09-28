# Formal-language experiments

One harness, every architecture, every task; the tasks are generated on the fly from the seed.

* `harness/train_statetrack.py`: the eager harness. `--arch dualmod | vanilla | lstm | gdn | looped | fbt`;
  `--task group` (A5 word problem, `--group a5 --format tagged`), `keyed` (keyed A5), `cvp` (circuit value),
  `dfa` (C-RASP length generalization, `--dfa z2|z5|z10|a5|flipflop|contains_a|contains_ab`), `parity`.
  Width presets `--scale 63m` (d=768, 9 layers, 12 heads) and `w1280` (d=1280); `--n_layers` overrides depth.
  Loop recipes: iteration-agnostic (`--loop_k K --loop_k_min K/4 --eval_loop_ks ...`), K proportional to length
  (`--loop_k_per_ops S`), recurrent-depth (`--loop_huginn --loop_emb_scale --loop_k 128`). FBT multi-pass training:
  `--fbt_sched "75:1,22:2,3:3"`. Warm starts: `--init_from <ckpt>` with `--K_curriculum`, `--key_shuffle`,
  `--newkey_init mean` (keyed register ramps) and `--curriculum_end`. `--eval_dump` saves the checkpoint and the
  per-position evaluation artifact; `--early_stop 0.995` stops at the first 1000-step eval above the threshold.
* `harness/train_statetrack_ws.py`: DM on the WaveScan engine (keyed and CVP), same flags, `build_dm()`; use it for
  any DM cell longer than a few hundred tokens (CVP at 64 gates: 2 h instead of 19 h).
* `harness/statetrack_gen.py`: generators, Cayley tables, the DFA library, and the CVP / keyed certification helpers.
* `loop_fixed_point/loop_fixed_point.py`: relative state change and readout accuracy at every iterate of a
  trained looped checkpoint, in- and out-of-distribution.

Defaults: 12k steps, batch 64, lr 1e-3 (every reported cell sets `--lr` explicitly: 4e-4 for DM / LSTM / FBT,
1e-4 for the tuned Transformer++ and the loops, 1e-3 for GDN on A5), seed 1337, T=64, extrapolation factors 1, 2, 4.
Each run prints one `FINAL <tag> <arch>: {...}` line with in-distribution accuracy, per-position (`x1_by_pos8`)
or per-depth (`x1_by_k`) bins, and every extrapolation factor; JSON copies land in `analysis/`.

| folder | script | reproduces |
|---|---|---|
| `a5/` | `run_a5_T64.sh <arm>`, `run_a5_minlayers.sh`, `run_a5_depth_heatmap.sh` | A5 table and position curves, minimum layers, length x depth heat map |
| `keyed_a5/` | `run_keyed.sh <mode>`, `run_dm_lineage.sh` | keyed A5 table: cold cells, DM register ramp to K=1024, recurrent-depth / GDN / FBT ramps |
| `cvp/` | `run_cvp.sh <mode>`, `run_dm_lineage.sh` | CVP table, accuracy by depth, minimum layers; DM chain to 1024 gates |
| `length_gen/` | `run_lengthgen.sh <arm> <language>`, `run_parity_control.sh` | C-RASP length generalization on seven languages; the looped-Transformer positive control |
| `loop_fixed_point/` | `run_fixed_points.sh <ckpt> <task>` | loop iterate traces behind the fixed-point figure and tables |

Set `PY_GDN` to an interpreter with `flash-linear-attention` for the `gdn` arms; `GPU=<i>` selects the device.
Approximate cost on one B300: A5 T=64 cells 10-30 min (FBT ~4.5 h, exact sequential training); CVP 30k-step
cells 0.3-1.5 h (DM on the engine: 0.7 h at 16 gates, 2 h at 64); keyed DM at K=1024: ~40 h; C-RASP cells 10-25 min.
