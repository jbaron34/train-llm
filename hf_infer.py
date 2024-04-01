import time
import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    MistralForCausalLM,
    GenerationConfig,
)


BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage="bfloat16",
)
MODEL = MistralForCausalLM.from_pretrained(
    "runs/merged",
    quantization_config=BNB_CONFIG,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
TOKENIZER = AutoTokenizer.from_pretrained("runs/merged")
GENERATION_CONFIG = GenerationConfig.from_pretrained("runs/merged")


def infer(messages):
    text = TOKENIZER.apply_chat_template(messages, return_tensors="pt", tokenize=False)
    inputs = TOKENIZER(text, return_tensors="pt").input_ids.to(device="cuda:0", dtype=torch.long)
    generated = MODEL.generate(
        inputs,
        max_new_tokens=100,
        top_k=50,
        top_p=0.95,
        generation_config=GENERATION_CONFIG,
    )
    response = TOKENIZER.batch_decode(generated, skip_special_tokens=False)[0]
    return response.split("<|im_start|>assistant")[-1][1:].split("<|im_end|>")[0]


if __name__ == "__main__":
    N = 100
    messages = [
        {
            "role": "system",
            "content": "You are an ai image generator that takes user requests and interprets and converts them to optimized stable diffusion prompts. Always specify if the image is a photograph/painting/digital art/etc.",
        },
        {
            "role": "user",
            "content": "a photo of donald trump doing a handstand, 32k, HD, best quality",
        },
    ]
    start = time.time()
    for i in range(N):
        print(infer(messages))
    print((time.time() - start) / N)
