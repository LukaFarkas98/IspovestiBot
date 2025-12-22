import torch
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "../data/confessions_Archive_100k_clean_TRAINING.jsonl"
RUN_NAME = f"mistral-confessions-lora-100k-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
OUTPUT_DIR = f"./{RUN_NAME}"

# -----------------------
# LOAD DATASET
# -----------------------
dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

# -----------------------
# TOKENIZER
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    labels = tokens["input_ids"].copy()
    labels = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in labels
    ]

    tokens["labels"] = labels
    return tokens

dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

# -----------------------
# QLORA CONFIG
# -----------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# REQUIRED FOR MISTRAL
model.config.pad_token_id = tokenizer.pad_token_id

# -----------------------
# LORA CONFIG
# -----------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------
# DATA COLLATOR
# -----------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -----------------------
# TRAINING
# -----------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=5e-5,
    num_train_epochs=1,
    fp16=True,
    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

# -----------------------
# SAVE
# -----------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
