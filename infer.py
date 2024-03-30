import time
import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    MistralForCausalLM,
    GenerationConfig,
)
from peft import PeftModel


BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage="bfloat16",
)
MODEL = MistralForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=BNB_CONFIG,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
MODEL.model.embed_tokens = torch.nn.modules.sparse.Embedding(
    num_embeddings=32008,
    embedding_dim=4096,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
)
MODEL.lm_head = torch.nn.Linear(4096, 32008, bias=False)
MODEL = PeftModel.from_pretrained(MODEL, "runs/mistral-7b-sft-lora-fsdp")
MODEL.to("cuda:0")
TOKENIZER = AutoTokenizer.from_pretrained("runs/mistral-7b-sft-lora-fsdp")
GENERATION_CONFIG = GenerationConfig(
    early_stopping=True,
    decoder_start_token_id=0,
    eos_token_id=32000,
    pad_token=32001,
)


def infer(messages):
    text = TOKENIZER.apply_chat_template(messages, return_tensors="pt", tokenize=False)
    inputs = TOKENIZER(text, return_tensors="pt").input_ids
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
    for i in range(100):
        print(infer(messages))
    print(time.time() - start)
