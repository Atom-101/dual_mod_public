"""Run the eval battery on a HuggingFace CausalLM (base Llama-3.1-8B) through the
SAME task glue as the DM path (lm/eval/run_lm_eval.py) — for G0.2 harness parity
and for the BASE column of the hero eval battery (plans/llam_8b_convert.md §6).

Only the model backend differs from run_lm_eval.py: lm-eval's native HFLM instead
of DualModEval. Task registration (local based_tasks), RULER metadata threading and
the RULER sample-cap monkeypatch are identical, so a parity pass validates our
tasks + glue on a known-good model.

  python lm/eval/run_hf_eval.py --model_path models/Llama-3.1-8B \
     --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq,sciq,lambada_openai \
     --batch_size auto --tag base_llama31
"""
import argparse, hashlib, json, os, sys, functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _cap_samples(orig, n, *a, **k):
    k["num_samples"] = min(k.get("num_samples", n), n)
    return orig(*a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Llama-3.1-8B")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num_fewshot", type=int, default=None)
    ap.add_argument("--ctx_len", type=int, default=None)
    ap.add_argument("--ruler_lengths", default=None)
    ap.add_argument("--ruler_samples", type=int, default=None)
    ap.add_argument("--tag", default="base_llama31")
    args = ap.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    max_len = args.ctx_len or 8448
    lm = HFLM(pretrained=args.model_path, dtype=args.dtype, device=args.device,
              batch_size=args.batch_size, max_length=max_len,
              trust_remote_code=False)

    metadata = None
    if args.ruler_lengths:
        metadata = {"max_seq_lengths": [int(x) for x in args.ruler_lengths.split(",")],
                    "tokenizer": args.model_path}
        if args.ruler_samples:
            import lm_eval.tasks.ruler.niah_utils as nu
            nu.generate_samples = functools.partial(_cap_samples, nu.generate_samples,
                                                    args.ruler_samples)

    from lm_eval.tasks import TaskManager
    _bt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "based_tasks")
    task_manager = (TaskManager(include_path=_bt, metadata=(metadata or {}))
                    if os.path.isdir(_bt) else None)

    results = lm_eval.simple_evaluate(model=lm, tasks=args.tasks.split(","),
                                      num_fewshot=args.num_fewshot,
                                      metadata=metadata, task_manager=task_manager,
                                      confirm_run_unsafe_code=True,
                                      limit=args.limit)
    h = hashlib.md5(f"{args.tasks}:{args.num_fewshot}:{args.ruler_lengths}:{args.ctx_len}".encode()).hexdigest()[:6]
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(repo, "analysis", f"hfeval_{args.tag}_{h}.json")
    with open(out, "w") as f:
        json.dump({"model": args.model_path, "tasks": args.tasks,
                   "num_fewshot": args.num_fewshot, "results": results["results"]},
                  f, indent=1, default=str)
    print(json.dumps(results["results"], indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
