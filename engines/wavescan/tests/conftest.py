import os

# must be set before cuBLAS init for torch.use_deterministic_algorithms (T-K8)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
