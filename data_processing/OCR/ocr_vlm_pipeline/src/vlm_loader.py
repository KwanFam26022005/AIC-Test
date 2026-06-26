from __future__ import annotations


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


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

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map=cfg.vlm.get("device_map", "auto"),
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    ).eval()
    return model, tokenizer

