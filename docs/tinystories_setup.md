# TinyStories ablation — model and training setup (for the paper; source models/dualmod/config.py, dualmod/train.py, run logs)
Data: TinyStories, GPT-2 BPE (vocab 50,257), packed 512-token sequences; train 474.0M tokens, val 4.77M tokens. Identical token stream for all arms (data_seed 42; epoch permutation, union over DDP ranks == single-process batch). Budget 300M tokens = 1,144 steps of 512 sequences x 512 tokens (262,144 tokens/step). Well under one epoch.
Trunk (all arms): pre-norm Transformer, 8 layers, RMSNorm, RoPE (theta 1e4), SwiGLU MLP with hidden = round(8/3 d) to a multiple of 64, tied input/output embeddings, no dropout, MHA (n_kv_heads = n_heads).
  width-matched TPP (A): d=512, 8 heads (head_dim 64), vanilla SDPA attention. 50.64M params.
  param-matched TPP (E): d=576, 9 heads (head_dim 64), vanilla SDPA. 60.81M params.
  DM (B): A's trunk + dual modulation in every layer (k-mod + v-mod), sequential exact scan (attn_mode=sequential; checkpoint_chunk 64). 63.24M params (+12.60M).
  v-mod only (C): 56.94M (+6.30M). k-mod only (D): 56.94M (+6.30M).
DM knobs in these runs: ctx_mod=linear (linear context transition), gate_type=convex_sigmoid (k' = g*k_raw + (1-g)*k_ctx), gate_type_v=convex, gate_input=xu (gate reads [x_hat; u]), rmsnorm_on_o, ctx_proj_init_std 0.006, fp32 softmax and fp32 gate sigmoid inside bf16 autocast, kmod_vraw_heads=0 (no reserved raw-value head), no LoRA factoring (gate_rank=ctx_rank=0), no v_clamp / k-norm.
  gate_bias_init = 0 for B/C/D here (gate starts fully mixed, sigma(0)=0.5). The original 2026-07 runs used gate_bias_init=4 (sigma(4)=0.98, near-vanilla start) and under that init the mechanism lost to width (RESULTS.md, superseding note); gate-0 reverses it.
Optimizer: AdamW(0.9, 0.95), weight decay 0.1 (none on norms, gate biases, embeddings), lr 3e-4 cosine to 3e-5, 1% linear warmup, grad clip 1.0, bf16 autocast. Seeds: model/training seed 1337 for A/E/B(B-ginit0); 2001 for TS0-C/D (data stream fixed regardless).
Eval: val loss on the held-out split every 100 steps (32 x 512-seq batches; 13 evals over the run), final = step 1144.
Hardware: 2 x B300 DDP per run (A/E/B: 2026-07-04; C/D: 2026-09-25, ~16 s/step). The 2026-09-25 relaunch of C/D on 1 GPU x bsz128 x accum 4 was 4x slower because the eager sequential scan is kernel-launch bound (killed; DDP rerun used).
Final val loss (single seed): A 1.7369 (TS0-A seeds 1.7338/1.7357/1.7332), E 1.6840 (seeds 1.6874/1.6931), C 1.7025, D 1.6969, B 1.6788.
