import torch
from transformers import (
    AutoTokenizer,
    MistralForCausalLM,
    GenerationConfig,
    MistralConfig,
)
from peft import PeftModel


model_config = MistralConfig(
    **{
        "_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "architectures": [
            "MistralForCausalLM"
        ],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "model_type": "mistral",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.39.1",
        "use_cache": True,
        "vocab_size": 32000
    }
)
model = MistralForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    config=model_config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
model.model.embed_tokens = torch.nn.modules.sparse.Embedding(
    num_embeddings=32008,
    embedding_dim=4096,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
)
model.lm_head = torch.nn.Linear(4096, 32008, bias=False)
model.model.vocab_size = 32008
model.config.vocab_size = 32008
model = PeftModel.from_pretrained(model, "runs/mistral-7b-sft-lora-fsdp")
model.to(dtype=torch.bfloat16)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("runs/merged")

tokenizer = AutoTokenizer.from_pretrained("runs/mistral-7b-sft-lora-fsdp")
tokenizer.save_pretrained("runs/merged")

GenerationConfig(
    do_sample=True,
    decoder_start_token_id=0,
    eos_token_id=32000,
    pad_token=32001,
).save_pretrained("runs/merged")
