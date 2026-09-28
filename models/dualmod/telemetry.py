"""§10 telemetry: gate statistics, register fractions, token-class breakdown,
attention mass onto registers, ctx/raw norm ratios. Runs on a fixed validation
batch every ~500 steps; scalars + histogram arrays are returned for the train
loop to push to wandb (and mirror to jsonl)."""

import numpy as np
import torch


@torch.no_grad()
def collect_telemetry(model, x_val, function_mask, prefix="telemetry"):
    """model: DualModLM in sequential/deq mode; x_val: [B, T] fixed val batch;
    function_mask: bool tensor [vocab] (§10.3 classes).

    Returns (scalars: dict[str, float], hists: dict[str, np.ndarray]).
    """
    was_training = model.training
    model.eval()
    collect = [dict() for _ in model.blocks]
    model(x_val, collect=collect)
    if was_training:
        model.train()

    T = x_val.shape[1]
    is_function = function_mask.to(x_val.device)[x_val]          # [B, T] bool
    scalars, hists = {}, {}

    for li, c in enumerate(collect):
        p = f"{prefix}/L{li}"
        for tag in ("k", "v"):
            if f"g_{tag}" not in c:
                continue
            g = torch.cat(c[f"g_{tag}"], dim=1)                  # [B, T, d]
            tok_mean = g.mean(dim=-1)                            # per-token mean over dims
            tok_min = g.min(dim=-1).values                       # per-token min over dims
            scalars[f"{p}/g_{tag}_mean"] = tok_mean.mean().item()
            scalars[f"{p}/g_{tag}_tokmin_mean"] = tok_min.mean().item()
            # §10.2 register fraction: tokens whose mean (1 - g) > 0.5
            scalars[f"{p}/register_frac_{tag}"] = ((1 - tok_mean) > 0.5).float().mean().item()
            # §10.3 token-class breakdown
            scalars[f"{p}/g_{tag}_function"] = tok_mean[is_function].mean().item()
            scalars[f"{p}/g_{tag}_content"] = tok_mean[~is_function].mean().item()
            hists[f"{p}/g_{tag}_hist"] = tok_mean.flatten().cpu().numpy()
            hists[f"{p}/g_{tag}_tokmin_hist"] = tok_min.flatten().cpu().numpy()
            # §10.5 norm ratios ||ctx|| / ||raw||
            ratio = torch.cat(c[f"ratio_{tag}"], dim=1)          # [B, T]
            scalars[f"{p}/norm_ratio_{tag}"] = ratio.mean().item()

        # §10.4 attention mass onto registers (uses key gate)
        if "inc_mass" in c and "g_k" in c:
            g_k_tok = torch.cat(c["g_k"], dim=1).mean(dim=-1)    # [B, T]
            denom = torch.arange(T, 0, -1, device=x_val.device).float()  # queries per key
            received = c["inc_mass"] / denom                     # [B, T]
            registerness = (1 - g_k_tok).flatten()
            rec = received.flatten()
            q10 = torch.quantile(registerness, 0.9)
            q90 = torch.quantile(registerness, 0.1)
            top, bot = registerness >= q10, registerness <= q90
            scalars[f"{p}/attn_mass_top_decile_1mgk"] = rec[top].mean().item()
            scalars[f"{p}/attn_mass_bot_decile_1mgk"] = rec[bot].mean().item()

    # cross-layer aggregates (handy single curves to watch)
    for tag in ("k", "v"):
        vals = [v for k_, v in scalars.items() if k_.endswith(f"/g_{tag}_mean")]
        if vals:
            scalars[f"{prefix}/g_{tag}_mean_all_layers"] = float(np.mean(vals))
        regs = [v for k_, v in scalars.items() if k_.endswith(f"/register_frac_{tag}")]
        if regs:
            scalars[f"{prefix}/register_frac_{tag}_all_layers"] = float(np.mean(regs))
    return scalars, hists


@torch.no_grad()
def grad_norm_ratios(model, prefix="telemetry"):
    """§10.5: grad-norm of W_Kctx/W_Vctx vs W_K/W_V per layer (call after backward,
    before optimizer.step / zero_grad)."""
    scalars = {}
    for li, blk in enumerate(model.blocks):
        a = blk.attn
        p = f"{prefix}/L{li}"
        pairs = []
        if hasattr(a, "w_kctx"):
            pairs.append((a.w_kctx, a.wk, "k"))
        if hasattr(a, "w_vctx"):
            pairs.append((a.w_vctx, a.wv, "v"))
        for ctx_w, raw_w, tag in pairs:
            if ctx_w.weight.grad is not None and raw_w.weight.grad is not None:
                g_ctx = ctx_w.weight.grad.norm().item()
                g_raw = raw_w.weight.grad.norm().item()
                scalars[f"{p}/gradnorm_w{tag}ctx"] = g_ctx
                scalars[f"{p}/gradnorm_ratio_{tag}ctx_over_raw"] = g_ctx / (g_raw + 1e-12)
    return scalars
