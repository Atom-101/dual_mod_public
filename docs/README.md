# Notes

* `fbt_analysis.md`, `fbt_flops_note.md`: the full-bandwidth Transformer baseline: the paper-faithful implementation,
  what exact sequential training reaches on every task, the multi-pass schedule sweep, and the training-cost model
  (FBT recomputes the whole stack per sweep, f = 1, against f ~ 0.1 for DM: 25x vs 3.3x at the 1.3B schedule).
* `loop_numbers.md`: every number behind the loop fixed-point figure and tables (relative state change per iterate,
  deepest-bin accuracy in- and out-of-distribution), plus the CVP per-recipe depths and the keyed recurrent-depth
  run log. Its Sec. 7 remark that the one-block recurrent-depth loop holds no keyed state predates the embedding-scale
  fix; with the scale it holds K=1024 (see the README corrections).
* `tinystories_setup.md`: the complete TinyStories ablation recipe.
* `gdn2_table.md`, `gdn2_vs_dm_results.md`: the 1.3B comparison protocol against Gated DeltaNet-2 (metric mapping,
  data recipe verdict, official-RULER vs lm-eval RULER, the K=24 vs K=64 parity check).
