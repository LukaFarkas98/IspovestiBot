import modal
from datetime import datetime
import shutil
import os

# -----------------------
# SETTINGS
# -----------------------
volume_name = "YUGOGPT_training_ispovesti_volume"
training_file = "confessions_for_training_YUGOGPT_cleaned.jsonl"
base_model = "gordicaleksa/YugoGPT"
max_len = 384

# -----------------------
# CREATE MODAL APP
# -----------------------
app = modal.App("YugoGPT-ISPOVESTI-QLORA-OPTIMIZED")

volume = modal.Volume.from_name(volume_name)

@app.function(
    gpu="A100-40GB",
    timeout=60 * 60 * 14,  # 14 hours instead of 24 to save cost
    memory=64000,
    volumes={"/mnt/models": volume},
    image=modal.Image.debian_slim().pip_install([
    "torch>=2.1.0",
    "transformers>=4.34.0",
    "datasets>=2.14.0",
    "peft>=0.5.0",
    "accelerate>=0.24.0",
    "bitsandbytes>=0.41.1",
    "sentencepiece"
])
)
def train_yugo_gpt():
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        Trainer, TrainingArguments, default_data_collator
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    # Paths inside volume
    DATA_PATH = f"/mnt/models/{training_file}"
    OUTPUT_DIR = f"/mnt/models/YugoGPT-confessions-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"Loading dataset from: {DATA_PATH}")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # -----------------------
    # Tokenizer
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length"
        )
        tokens["labels"] = [t if t != tokenizer.pad_token_id else -100 for t in tokens["input_ids"]]
        return tokens

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"], desc="Tokenizing")

    # -----------------------
    # QLoRA 4-bit Config
    # -----------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": 0}
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # -----------------------
    # LoRA Config
    # -----------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------
    # Trainer
    # -----------------------
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            num_train_epochs=1,  # optimized: start with 1 epoch first
            fp16=True,
            logging_steps=200,
            save_steps=2000,
            save_total_limit=2,
            report_to="none"
        ),
        train_dataset=dataset,
        data_collator=default_data_collator
    )

    print("Starting training...")
    trainer.train()

    # -----------------------
    # Save artifacts
    # -----------------------
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Training complete. Model saved at {OUTPUT_DIR}")

    # -----------------------
    # Cleanup
    # -----------------------
    for item in os.listdir("/mnt/models/"):
        path = os.path.join("/mnt/models/", item)
        if path not in {DATA_PATH, OUTPUT_DIR}:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

# -----------------------
# LOCAL ENTRYPOINT
# -----------------------
@app.local_entrypoint()
def main():
    print("Deploying Modal app and starting training...")
    app.deploy()
    run_id = train_yugo_gpt.spawn()
    print("Training started! Use `modal logs {run_id}` to watch.")
