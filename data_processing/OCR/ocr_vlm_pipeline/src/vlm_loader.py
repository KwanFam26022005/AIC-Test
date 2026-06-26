from __future__ import annotations


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
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map=cfg.vlm.get("device_map", "auto"),
        low_cpu_mem_usage=True,
    ).eval()
    return model, tokenizer

