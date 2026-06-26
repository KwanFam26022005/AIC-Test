from __future__ import annotations

from contextlib import contextmanager


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _install_flash_attn_stub() -> None:
    """Let remote VLM code import flash_attn while using eager attention.

    Some Vintern/InternVL modeling files import flash_attn at module import time
    even when `attn_implementation="eager"` is requested. Building flash-attn on
    Colab T4/Python 3.12 is fragile, so this stub satisfies import checks. If a
    model actually tries to call FlashAttention kernels, the stub raises a clear
    error instead of silently producing wrong results.
    """
    import importlib.machinery
    import sys
    import types

    if "flash_attn" in sys.modules:
        return

    def _unavailable(*args, **kwargs):
        raise RuntimeError("flash_attn is not installed; set attn_implementation='eager'.")

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_attn.flash_attn_func = _unavailable
    flash_attn.flash_attn_varlen_func = _unavailable

    flash_attn_interface = types.ModuleType("flash_attn.flash_attn_interface")
    flash_attn_interface.__spec__ = importlib.machinery.ModuleSpec("flash_attn.flash_attn_interface", loader=None)
    flash_attn_interface.flash_attn_func = _unavailable
    flash_attn_interface.flash_attn_varlen_func = _unavailable

    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    bert_padding.index_first_axis = _unavailable
    bert_padding.pad_input = _unavailable
    bert_padding.unpad_input = _unavailable

    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface
    sys.modules["flash_attn.bert_padding"] = bert_padding


@contextmanager
def _no_quantized_model_to_call(enabled: bool):
    if not enabled:
        yield
        return

    import transformers.modeling_utils

    original_to = transformers.modeling_utils.PreTrainedModel.to

    def _safe_to(self, *args, **kwargs):
        try:
            return original_to(self, *args, **kwargs)
        except ValueError as exc:
            message = str(exc)
            if "4-bit" in message or "8-bit" in message or "bitsandbytes" in message:
                return self
            raise

    transformers.modeling_utils.PreTrainedModel.to = _safe_to
    try:
        yield
    finally:
        transformers.modeling_utils.PreTrainedModel.to = original_to


def load_transformers_vlm(model_id: str, cfg):
    import torch
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    quantization = cfg.vlm.get("quantization", "4bit_nf4")
    quant_config = None
    if quantization == "4bit_nf4":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Use flash_attention_2 when flash_attn is installed, otherwise fall back
    # to the default eager implementation to avoid ImportError on Colab/CPU.
    attn_impl = cfg.vlm.get("attn_implementation", None)
    if attn_impl is None:
        attn_impl = "flash_attention_2" if _has_flash_attn() else "eager"
    if attn_impl == "eager" and cfg.vlm.get("stub_flash_attn", True) and not _has_flash_attn():
        _install_flash_attn_stub()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    patch_to = bool(quant_config is not None and cfg.vlm.get("patch_quantized_to", True))
    with _no_quantized_model_to_call(patch_to):
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map=cfg.vlm.get("device_map", "auto"),
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        ).eval()
    return model, tokenizer
