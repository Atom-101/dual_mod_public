"""Multi-GPU equivalence test for the fused-sweep engine's DDP path (§6):

  torchrun --nproc_per_node=N engines/wavescan/tests/ddp_equiv.py

1. grad equivalence: N-rank sharded train_step + finish_grads(world_size)
   all-reduce must equal a single-process full-batch run (fp32, 1e-5).
2. trajectory equivalence: 20 AdamW steps, DDP loss curve == single-process
   full-batch loss curve (fp32; bitwise-grade).
3. bf16 smoke: 5 steps, finite loss, ranks stay weight-synced.
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.dualmod.config import DualModConfig
from models.dualmod.model import DualModLM
from engines.wavescan.engine.graphs import WaveScanEngine
from engines.wavescan.engine.schedule import uniform_chunks


def relerr(a, b):
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")

    torch.manual_seed(0)                      # identical init on all ranks
    cfg = DualModConfig(d_model=128, n_layers=2, n_heads=4, vocab_size=512,
                        max_seq_len=64, gate_bias_init=0.0, checkpoint_chunk=0,
                        ctx_proj_init_std=0.02)
    model = DualModLM(cfg).cuda().float().eval()
    init_sd = {k: v.clone() for k, v in model.state_dict().items()}
    Bs, T = 4, 64
    sched = [uniform_chunks(T, 16, 4)]
    g = torch.Generator().manual_seed(7)
    steps = 20
    data = [(torch.randint(0, cfg.vocab_size, (world * Bs, T), generator=g).cuda(),
             torch.randint(0, cfg.vocab_size, (world * Bs, T), generator=g).cuda())
            for _ in range(steps)]

    # ---- single-process full-batch reference (rank 0) ----------------------
    ref_losses, ref_grads = [], None
    if rank == 0:
        eng_ref = WaveScanEngine(model, sched, world * Bs, T, use_graphs=True,
                                 fused_sweep=True)
        loss = eng_ref.train_step(data[0][0], data[0][1], 0).clone()
        eng_ref.finish_grads()
        ref_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        ref_loss0 = loss.item()
        # trajectory
        model.load_state_dict(init_sd)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for s in range(steps):
            l = eng_ref.train_step(data[s][0], data[s][1], 0)
            ref_losses.append(l.item())
            eng_ref.finish_grads()
            opt.step()
            opt.zero_grad(set_to_none=False)
        model.load_state_dict(init_sd)
        model.zero_grad(set_to_none=True)
    dist.barrier()
    # re-sync weights (rank 0 mutated + restored; broadcast for safety)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # ---- DDP sharded engines ------------------------------------------------
    eng = WaveScanEngine(model, sched, Bs, T, use_graphs=True,
                         fused_sweep=True, cap_priority=0 if rank == 0 else -1)
    sh = slice(rank * Bs, (rank + 1) * Bs)

    loss = eng.train_step(data[0][0][sh], data[0][1][sh], 0).clone()
    dist.all_reduce(loss)
    loss /= world
    eng.finish_grads(world_size=world)
    if rank == 0:
        worst, wname = 0.0, ""
        for n, p in model.named_parameters():
            e = relerr(p.grad, ref_grads[n])
            if e > worst:
                worst, wname = e, n
        print(f"[grad-equiv] loss ddp={loss.item():.6f} ref={ref_loss0:.6f} "
              f"dloss={abs(loss.item()-ref_loss0):.2e}  worst grad {wname} "
              f"{worst:.2e}", flush=True)
        assert abs(loss.item() - ref_loss0) < 1e-5 and worst < 1e-5, "GRAD MISMATCH"

    # ---- trajectory ----------------------------------------------------------
    model.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    traj = []
    for s in range(steps):
        l = eng.train_step(data[s][0][sh], data[s][1][sh], 0).clone()
        dist.all_reduce(l)
        traj.append((l / world).item())
        eng.finish_grads(world_size=world)
        opt.step()
        opt.zero_grad(set_to_none=False)
    if rank == 0:
        worst = max(abs(a - b) for a, b in zip(traj, ref_losses))
        print(f"[trajectory] {steps} steps, max |Δloss| = {worst:.2e}", flush=True)
        print(f"  ddp: {[f'{x:.4f}' for x in traj[:5]]} ... {traj[-1]:.4f}", flush=True)
        print(f"  ref: {[f'{x:.4f}' for x in ref_losses[:5]]} ... {ref_losses[-1]:.4f}", flush=True)
        assert worst < 1e-4, "TRAJECTORY MISMATCH"

    # ---- bf16 smoke ----------------------------------------------------------
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    eng16 = WaveScanEngine(model, sched, Bs, T, use_graphs=True,
                           fused_sweep=True, autocast_bf16=True,
                           cap_priority=0 if rank == 0 else -1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for s in range(5):
        l = eng16.train_step(data[s][0][sh], data[s][1][sh], 0).clone()
        eng16.finish_grads(world_size=world)
        opt.step()
        opt.zero_grad(set_to_none=False)
        assert torch.isfinite(l).all()
    # weight sync check across ranks: BITWISE compare vs rank-0 bytes.
    # (all_reduce(mean)-based checks are wrong: summing N identical fp32
    # values hits non-power-of-2 intermediates that round — one ulp of pure
    # measurement noise at N>2.)
    drift = torch.zeros((), device="cuda")
    for p in model.parameters():
        pm = p.data.clone()
        dist.broadcast(pm, src=0)
        drift = torch.maximum(drift, (pm != p.data).any().float())
    dist.all_reduce(drift, op=dist.ReduceOp.MAX)
    d = drift.item()
    if rank == 0:
        print(f"[bf16-smoke] 5 steps ok, max cross-rank weight drift = {d:.2e}",
              flush=True)
    assert d == 0.0, f"rank desync: {d}"
    if rank == 0:
        print("DDP EQUIVALENCE: ALL PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
