from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, GenerationConfig


QUANT_CONFIG = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

MODEL = AutoAWQForCausalLM.from_pretrained("runs/merged", low_cpu_mem_usage=True, device_map="cpu")
TOKENIZER = AutoTokenizer.from_pretrained("runs/merged", trust_remote_code=True)
GENERATION_CONFIG = GenerationConfig.from_pretrained("runs/merged")


MODEL.quantize(TOKENIZER, quant_config=QUANT_CONFIG)

MODEL.save_quantized("runs/quantized")
TOKENIZER.save_pretrained("runs/quantized")
GENERATION_CONFIG.save_pretrained("runs/quantized")