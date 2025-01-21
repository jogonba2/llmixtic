from typing import Final

import torch
from transformers import BitsAndBytesConfig

QUANTIZATION_CONFIGS: Final = {
    "int4": {
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        "low_cpu_mem_usage": True,
    },
    "int4_bf16": {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        ),
        "low_cpu_mem_usage": True,
    },
    "int4_nf4": {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4"
        ),
        "low_cpu_mem_usage": True,
    },
    "int8": {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)},
    "int4_nested": {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True
        ),
        "low_cpu_mem_usage": True,
    },
    "fp16": {"torch_dtype": torch.float16},
    "none": {},
}
