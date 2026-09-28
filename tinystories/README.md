# TinyStories k-mod / v-mod ablation

Five 8-layer arms on one shared 300M-token TinyStories stream (GPT-2 BPE, 512-token sequences, batch 512):
width-matched Transformer++ (d=512, 50.6M), param-matched Transformer++ (d=576, 60.8M), DM (63.2M), v-mod only
(56.9M) and k-mod only (56.9M). The DM arms run the exact eager sequential scan with gate bias 0 (fully mixed
start). Final validation losses: 1.737 / 1.684 / **1.679** / 1.702 / 1.697 (paper Table "tinystories",
Fig. "tinystories").

```bash
python -m models.dualmod.data --data_dir data                 # TinyStories -> data/{train,val}.bin (~474M / 4.8M tokens)
ARM=A GPUS=0,1 bash tinystories/run_ablation.sh              # then E, B, C, D; 2-GPU DDP, ~5 h per DM arm on B300
python figures/render/render_tinystories.py                  # reads out/<run>/metrics.jsonl
```

`train.py` is the small self-contained trainer (`--preset A|B|C|D`, every `TrainConfig` / `DualModConfig` field is a
flag); see `docs/tinystories_setup.md` for every hyper-parameter.
