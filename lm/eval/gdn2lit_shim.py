"""Import shim so the official GDN-2 lit_gpt code loads in our venv without flash_attn (pure GDN-2 never
calls attention). Import this BEFORE lit_gpt."""
import sys, types, importlib.machinery as im
if "flash_attn" not in sys.modules:
    fa = types.ModuleType("flash_attn"); fa.__spec__ = im.ModuleSpec("flash_attn", None); fa.__version__ = "2.7.4"
    def _no(*a, **k): raise RuntimeError("flash_attn shim: an attention layer was called (gdn2_per_layer must be 1)")
    fa.flash_attn_func = fa.flash_attn_varlen_func = _no
    bp = types.ModuleType("flash_attn.bert_padding"); bp.__spec__ = im.ModuleSpec("flash_attn.bert_padding", None)
    bp.index_first_axis = bp.pad_input = bp.unpad_input = _no; fa.bert_padding = bp
    sys.modules["flash_attn"] = fa; sys.modules["flash_attn.bert_padding"] = bp
import os
GDN2_ROOT = os.environ.get("GDN2_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "gateddeltanet-2"))
if GDN2_ROOT not in sys.path:
    sys.path.insert(0, GDN2_ROOT)
