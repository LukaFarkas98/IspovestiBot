import modal
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, PeftModel
from peft import prepare_model_for_kbit_training
import os, shutil

# -----------------------
# Modal App
# -----------------------
app = modal.App("mistral-confessions")

# Persistent volume (dataset stays, everything else can be cleaned)
volume = modal.Volume.from_name("mistral_models")

# -----------------------
# Remote training function
# -----------------------
@app.function(
    gpu="A100-40GB",
    timeout=60 * 60 * 24,
    memory=64000,
    volumes={"/mnt/models": volume},
    image=modal.Image.debian_slim().pip_install([
        "torch", "transformers", "datasets", "peft", "accelerate", "bitsandbytes"
    ]),
)
def train_confessions_model():
    OUTPUT_DIR = f"/mnt/models/mistral-confessions-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    DATA_PATH = "/mnt/models/confessions_Archive_100k_clean_TRAINING.jsonl"

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )
        tokens["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in tokens["input_ids"]]
        ][0]
        return tokens

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            num_train_epochs=2,
            fp16=True,
            logging_steps=200,
            save_steps=2000,
            save_total_limit=2,
            report_to="none",
        ),
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Training finished. Model saved at {OUTPUT_DIR}")

    # cleanup
    for item in os.listdir("/mnt/models/"):
        path = os.path.join("/mnt/models/", item)
        if path != DATA_PATH and path != OUTPUT_DIR:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

# -----------------------
# Local entrypoint
# -----------------------
@app.local_entrypoint()
def main():
    print("Deploying App and starting detached training...")
    # Step 1: deploy App
    app.deploy()
    # Step 2: spawn detached training
    run_name = train_confessions_model.spawn()
    print("Training job started in detached mode. Run ID:", run_name)
    print("Check logs with: modal logs", run_name)
