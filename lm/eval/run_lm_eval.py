"""Run lm-eval-harness tasks on a DualModLM checkpoint.

fla-seven (likelihood protocol):
  python lm/eval/run_lm_eval.py --ckpt out/FS-DM-s1337/ckpt_latest.pt \
    --tasks piqa,hellaswag,winogrande,arc_easy,arc_challenge,lambada_openai,wikitext
Writes analysis/lmeval_<run>_<step>_<tasks-hash>.json
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _cap_samples(orig, n, *a, **k):
    k["num_samples"] = min(k.get("num_samples", n), n)
    return orig(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--batch_rows", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tok_dir", default="data/llama2_tok")
    ap.add_argument("--num_fewshot", type=int, default=None)
    ap.add_argument("--ctx_len", type=int, default=None,
                    help="context extension (RULER extrapolation)")
    ap.add_argument("--ruler_lengths", default=None,
                    help="comma seq lengths for RULER synthetic tasks")
    ap.add_argument("--log_samples", action="store_true")
    ap.add_argument("--ruler_samples", type=int, default=None,
                    help="override RULER's hardcoded 500 samples/length")
    ap.add_argument("--engine_K", type=int, default=None,
                    help="score loglikelihood tasks through the wave3d training engine at this K "
                         "(64 = exact at C=64; 24 = final training operator). Generation stays exact-decode.")
    ap.add_argument("--T_eval", type=int, default=2048,
                    help="engine sequence length (multiple of 64); rows are right-padded to it")
    ap.add_argument("--engine_kwargs", default="{}", help="json kwargs for the wave3d engine")
    ap.add_argument("--engine_compile", default="all")
    ap.add_argument("--gen_rows", type=int, default=4, help="engine generation batch rows (engine_K mode)")
    ap.add_argument("--tag", default=None, help="output-name tag (also used to find the job via pgrep)")
    ap.add_argument("--graph_decode", action="store_true",
                    help="generation via the CUDA-graph static-cache decoder (validate with lm/eval/validate_graph_decode.py first)")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="data-parallel: split each task's docs into N disjoint shards")
    ap.add_argument("--shard_id", type=int, default=0)
    args = ap.parse_args()

    import lm_eval
    from lm.eval.lm_eval_adapter import DualModEval, Wave3DEval
    import torch
    import torch.distributed as dist

    # torchrun data-parallel: every rank builds the model; lm-eval shards requests by
    # rank and gathers on rank 0 (adapter hooks). Single-process runs are unchanged.
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(hours=12))   # generation is rank-imbalanced: fast ranks wait at the barrier
    rank = dist.get_rank() if distributed else 0

    if args.engine_K and os.environ.get("EVAL_ARCH") in ("gdn2lit", "gdn"):
        # the wave3d engine is DM-only; third-party GDN-2 ckpts run the eager (DualModEval) path.
        # engine_K stays in the tag/hash for phase bookkeeping but is not an operator here.
        if rank == 0:
            print(f"[run_lm_eval] EVAL_ARCH={os.environ['EVAL_ARCH']}: ignoring --engine_K {args.engine_K} (eager path)")
        args.engine_K = None
    if args.engine_K:
        lm = Wave3DEval(args.ckpt, tok_dir=args.tok_dir, batch_rows=args.batch_rows,
                        T_eval=args.T_eval, K=args.engine_K, ctx_len=args.ctx_len,
                        engine_kwargs=json.loads(args.engine_kwargs),
                        compile_mode=args.engine_compile, gen_rows=args.gen_rows)
    else:
        lm = DualModEval(args.ckpt, batch_rows=args.batch_rows, device=args.device,
                         tok_dir=args.tok_dir, ctx_len=args.ctx_len)
        lm.gen_rows = args.gen_rows
    if args.graph_decode:
        lm.graph_decode = True
    metadata = None
    if args.ruler_lengths:
        metadata = {"max_seq_lengths": [int(x) for x in args.ruler_lengths.split(",")],
                    "tokenizer": args.tok_dir}
        if args.ruler_samples:
            import functools
            import lm_eval.tasks.ruler.niah_utils as nu
            orig = nu.generate_samples
            nu.generate_samples = functools.partial(_cap_samples, orig,
                                                    args.ruler_samples)
            # lm-eval imports the task module from its file path (a fresh module object that binds
            # generate_samples at import time), so the cap must live on the source module too.
            import lm_eval.tasks.ruler.prepare_niah as pn
            pn.generate_samples = nu.generate_samples
    # register the local Based recall tasks (based_triviaqa/based_drop) alongside
    # the built-ins so --tasks can name them; include_defaults keeps everything else
    # Register the local Based recall tasks AND carry the metadata: simple_evaluate
    # only injects metadata (RULER's tokenizer) when it builds the TaskManager itself
    # (evaluator.py: `if task_manager is None: task_manager = TaskManager(metadata=...)`).
    # A custom task_manager skips that, so we must pass metadata in here — else RULER
    # fails "No tokenizer or pretrained provided". metadata=None (non-RULER) is harmless.
    from lm_eval.tasks import TaskManager
    _bt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "based_tasks")
    task_manager = (TaskManager(include_path=_bt, metadata=(metadata or {}))
                    if os.path.isdir(_bt) else None)
    tasks_arg = args.tasks.split(",")
    if args.num_shards > 1:
        # data-parallel sharding: slice each task's HF dataset into disjoint shards
        # (strided so each shard is size-balanced), then hand simple_evaluate the
        # pre-built + sharded task_dict. exact_match aggregates externally by mean.
        from lm_eval.tasks import get_task_dict
        td = get_task_dict(tasks_arg, task_manager)
        def _shard_task(t):
            cfg = getattr(t, "config", None)
            split = (getattr(cfg, "test_split", None) or getattr(cfg, "validation_split", None)) if cfg else None
            ds = getattr(t, "dataset", None)
            if split and ds is not None and hasattr(ds, "__contains__") and split in ds:
                ds[split] = ds[split].shard(num_shards=args.num_shards,
                                            index=args.shard_id, contiguous=False)
        def _walk(d):
            for v in d.values():
                _walk(v) if isinstance(v, dict) else _shard_task(v)
        _walk(td)
        tasks_arg = td
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks_arg,
                                      num_fewshot=args.num_fewshot,
                                      metadata=metadata,
                                      task_manager=task_manager,
                                      log_samples=args.log_samples,
                                      limit=args.limit)
    if distributed and rank != 0:          # lm-eval returns results only on rank 0
        dist.barrier()
        dist.destroy_process_group()
        return
    step = torch.load(args.ckpt, map_location="cpu",
                      weights_only=False).get("step", -1)
    run = os.path.basename(os.path.dirname(args.ckpt))
    h = hashlib.md5(f"{args.tasks}:{args.num_fewshot}:{args.ruler_lengths}:{args.ctx_len}:{args.engine_K}".encode()).hexdigest()[:6]
    if args.tag:
        h = f"{args.tag}_{h}"
    if args.engine_K:
        h = f"K{args.engine_K}_{h}"
    # EVAL_TAG: optional suffix so env-varied runs of the same task (JRT_NO_STRIP,
    # EVAL_PREFIX_*, EVAL_LEN_QUANT) do not overwrite each other's json/TEXT dumps.
    if os.environ.get("EVAL_TAG"):
        h += "_" + os.environ["EVAL_TAG"]
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    shard_sfx = f"_s{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
    out = os.path.join(repo, "analysis", f"lmeval_{run}_{step}_{h}{shard_sfx}.json")
    with open(out, "w") as f:
        payload = {"ckpt": args.ckpt, "step": step, "tasks": args.tasks,
                   "limit": args.limit, "engine_K": args.engine_K, "T_eval": args.T_eval,
                   "world_size": int(os.environ.get("WORLD_SIZE", "1")), "graph_decode": args.graph_decode,
                   "results": results["results"]}
        if args.log_samples:
            payload["samples"] = results.get("samples")
        json.dump(payload, f, indent=1, default=str)
    print(json.dumps(results["results"], indent=1, default=str))

    # READABLE full-text dump (one file per task): the exact prompt fed, the model's
    # generation, the gold target, and hit/miss — so failures are inspectable directly.
    if args.log_samples and results.get("samples"):
        def _prompt(s):
            a = s.get("arguments")
            try:                                    # generate_until: ((ctx, gen_kwargs),)
                return a[0][0] if isinstance(a[0], (list, tuple)) else a[0]
            except Exception:
                return str(a)
        def _gen(s):
            g = s.get("filtered_resps") or s.get("resps")
            while isinstance(g, (list, tuple)) and g:
                g = g[0]
            return g
        for task, samples in results["samples"].items():
            dpath = os.path.join(repo, "analysis", f"lmeval_{run}_{step}_{h}_{task}_TEXT.jsonl")
            n_hit = 0
            with open(dpath, "w") as df:
                for s in samples:
                    gen = _gen(s)
                    tgt = s.get("target")
                    metr = {k: v for k, v in s.items()
                            if k in ("contains", "exact_match", "f1", "em", "acc")}
                    hit = metr.get("contains", metr.get("exact_match", 0))
                    n_hit += 1 if (isinstance(hit, (int, float)) and hit >= 0.5) else 0
                    df.write(json.dumps({"prompt": _prompt(s), "gen": gen,
                                         "target": tgt, "metric": metr}, default=str) + "\n")
            print(f"[text dump] {task}: {len(samples)} docs, "
                  f"{n_hit} hits -> {dpath}")
    print("wrote", out)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
