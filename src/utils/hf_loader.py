from __future__ import annotations
import os
import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


def select_device(prefer_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(dtype_name: str = "float32", device: str = "cuda") -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.float16


def load_causal_lm(
    model_name: str,
    dtype: str = "float32",
    device: Optional[str] = None,
    output_hidden_states: bool = False,
    trust_remote_code: bool = True,
    load_kwargs: Optional[dict] = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:

    device = device or select_device()
    torch_dtype = select_dtype(dtype, device)
    load_kwargs = dict(load_kwargs or {})

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )

    if output_hidden_states:
        model.config.output_hidden_states = True
    if "device_map" not in load_kwargs:
        model = model.to(device)
    model.eval()
    return tokenizer, model, device
