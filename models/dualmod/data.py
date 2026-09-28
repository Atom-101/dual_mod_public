"""TinyStories data pipeline (plan.md §8).

`python -m models.dualmod.data --data_dir data` downloads roneneldan/TinyStories,
tokenizes with the GPT-2 BPE (tiktoken), packs each split into a flat uint16
memmap (stories separated by <|endoftext|>), and writes meta.pkl with token
frequencies + the "function token" mask used by §10.3 telemetry.

Batching: the token stream is cut into non-overlapping seq_len chunks; each
epoch visits all chunks in a permutation drawn from a fixed data_seed, so runs
A/B/C see the *identical* token stream.
"""

import argparse
import os
import pickle

import numpy as np
import torch


def prepare(data_dir: str, num_proc: int = 32):
    from datasets import load_dataset
    import tiktoken
    from tqdm import tqdm

    os.makedirs(data_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("roneneldan/TinyStories", num_proc=num_proc)

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    tokenized = ds.map(process, remove_columns=["text"], num_proc=num_proc,
                       desc="tokenizing")

    counts = np.zeros(enc.n_vocab, dtype=np.int64)
    for split, dset in tokenized.items():
        split_name = "val" if split == "validation" else split
        arr_len = int(np.sum(dset["len"], dtype=np.uint64))
        path = os.path.join(data_dir, f"{split_name}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(arr_len,))
        n_shards = 1024 if split == "train" else 32
        idx = 0
        for i in tqdm(range(n_shards), desc=f"writing {path}"):
            shard = tokenized[split].shard(num_shards=n_shards, index=i,
                                           contiguous=True).with_format("numpy")
            buf = np.concatenate(shard["ids"])
            arr[idx:idx + len(buf)] = buf
            idx += len(buf)
            if split == "train":
                counts += np.bincount(buf, minlength=enc.n_vocab)
        arr.flush()
        print(f"{path}: {arr_len:,} tokens")

    # §10.3 token classes: "function" = top-100 most frequent ∪ punctuation/whitespace
    top100 = np.argsort(counts)[::-1][:100]
    punct_ws = np.zeros(enc.n_vocab, dtype=bool)
    for tid in range(enc.n_vocab):
        try:
            s = enc.decode([tid])
        except Exception:
            continue
        if len(s) > 0 and not any(c.isalnum() for c in s):
            punct_ws[tid] = True
    function_mask = punct_ws.copy()
    function_mask[top100] = True
    meta = {"vocab_size": enc.n_vocab, "counts": counts, "top100": top100,
            "function_mask": function_mask}
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"meta.pkl written; {function_mask.sum()} function-class token ids")


class PackedLoader:
    """Deterministic loader over a packed .bin (identical stream for all runs)."""

    def __init__(self, data_dir: str, split: str, seq_len: int, batch_size: int,
                 data_seed: int = 42, device: str = "cuda",
                 rank: int = 0, world_size: int = 1):
        self.arr = np.memmap(os.path.join(data_dir, f"{split}.bin"),
                             dtype=np.uint16, mode="r")
        self.T = seq_len
        self.B = batch_size                       # GLOBAL batch (summed over ranks)
        assert batch_size % world_size == 0, "batch_size must be divisible by world_size"
        self.Bl = batch_size // world_size        # this rank's local batch
        self.rank = rank
        self.world_size = world_size
        self.seed = data_seed
        self.device = device
        self.n_chunks = (len(self.arr) - 1) // seq_len  # need T+1 tokens per chunk
        self._perm_epoch = -1
        self._perm = None

    def _epoch_perm(self, epoch: int) -> torch.Tensor:
        if epoch != self._perm_epoch:
            g = torch.Generator().manual_seed(self.seed + epoch)
            self._perm = torch.randperm(self.n_chunks, generator=g)
            self._perm_epoch = epoch
        return self._perm

    def _gather(self, chunk_ids) -> tuple[torch.Tensor, torch.Tensor]:
        xs = np.stack([self.arr[c * self.T: c * self.T + self.T + 1]
                       for c in chunk_ids]).astype(np.int64)
        t = torch.from_numpy(xs)
        x, y = t[:, :-1], t[:, 1:]
        if "cuda" in str(self.device):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def train_batch(self, step: int):
        """This rank's slice of the global batch for `step` (deterministic; permuted
        chunks, epoch-wise). Union over ranks == the single-process global batch, so
        the token stream is identical regardless of world_size."""
        ids = []
        for i in range(self.rank * self.Bl, (self.rank + 1) * self.Bl):
            s = step * self.B + i                 # index into the GLOBAL batch
            epoch, pos = divmod(s, self.n_chunks)
            ids.append(int(self._epoch_perm(epoch)[pos]))
        return self._gather(ids)

    def val_batch(self, i: int):
        """i-th fixed validation batch (sequential chunks, no shuffling); this rank's
        local slice."""
        base = (i * self.B) % max(self.n_chunks - self.B, 1)
        start = base + self.rank * self.Bl
        return self._gather(range(start, start + self.Bl))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--num_proc", type=int, default=32)
    args = ap.parse_args()
    prepare(args.data_dir, args.num_proc)
