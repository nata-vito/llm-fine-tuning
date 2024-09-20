from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)

from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

import huggingface_hub
import os, torch, wandb


def fine_tunning():
    # Login on platforms
    hf_token = os.environ.get('HF_TOKEN')
    wb_token = os.environ.get('WB_TOKEN')

    huggingface_hub.login(token = hf_token)
    wandb.login(key = wb_token)

    # start a new wandb run to track this script
    run = wandb.init(
        project = "llama-fine-tuning",
        job_type = "training",
        anonymous = "allow"
    )
    
    base_model = "/kaggle/input/llama-3/transformers/8b-chat-hf/1"
    dataset_name = "ruslanmv/ai-medical-chatbot"
    new_model = "llama-3-8b-chat-doctor"
    
    torch_dtype = torch.float16
    attn_implementation = "eager"
    
    # QLoRa config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch_dtype,
        bnb_4bit_use_double_quant = True,
    )

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config = bnb_config,
        device_map = "auto",
        attn_implementation = attn_implementation
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model, tokenizer = setup_chat_format(model, tokenizer)
    
    # LoRA config
    peft_config = LoraConfig(
        r = 16,
        lora_alpha = 32,
        lora_dropout = 0.05,
        bias = "none",
        task_type = "CAUSUAL_LM",
        target_modules = ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )
    
    model = get_peft_model(model, peft_config)


if __name__ == "__main__":
    fine_tunning()