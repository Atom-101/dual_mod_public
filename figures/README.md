# Figures

| figure | renderer | inputs (`figures/data/`) |
|---|---|---|
| `headline_right_keyed_cvp.png` (Fig. 1 right) | `render/render_headline_right.py` | numbers inline |
| `a5_T64_position_curves.png` | `render/render_a5_position.py` | A5 result JSONs and FINAL lines |
| `a5_depth_heatmap.png` | `render/render_a5_heatmap.py` | `lr1e4/`, `depth_sweep/` FINAL lines |
| `cvp_by_depth.png` | `render/render_cvp_by_depth.py` | CVP result JSONs and chain logs |
| `cvp_minlayers.png` | `render/render_cvp_minlayers.py` | numbers inline |
| `loop_fixedpoints_3x3_deepest.png` (paper), `loop_fixedpoints_3x3.png` | `render/render_loop_fixedpoints_3x3.py [--deepest]` | `loop_fixed_point/ood/*.txt` traces |
| `crasp_lengthgen.png` | `render/render_crasp_lengthgen.py` | `crasp/`, `looplr_dfa/`, `fbt/`, `prung_b300/` FINAL lines |
| `tinystories_ablation.png` | `render/render_tinystories.py` | `tinystories/<run>/metrics.jsonl` (val rows) |

Run any renderer from the repository root: `python figures/render/<name>.py`. The log inputs are trimmed to the
`FINAL` line each renderer parses and the JSONs to their `final` block; regenerating from this directory
reproduces the shipped PNGs pixel for pixel. The two architecture diagrams of Fig. 1 (left) are not in the repository.
