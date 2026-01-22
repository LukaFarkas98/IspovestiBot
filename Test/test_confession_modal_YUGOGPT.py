import modal
import json
import os

# -----------------------
# SETTINGS
# -----------------------
VOLUME_NAME = "YUGOGPT_training_ispovesti_volume"

# CHANGE THIS after training finishes
MODEL_DIR = "/mnt/models/YugoGPT-confessions-20260121-164854"

INPUT_FILE = "/mnt/models/test_yugo_prompts.txt"
OUTPUT_FILE = "/mnt/models/generated_confessions.json"

MAX_NEW_TOKENS = 180  # keeps confessions medium length

# -----------------------
# MODAL APP
# -----------------------
app = modal.App("YugoGPT-ISPOVESTI-INFERENCE")

volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(
    gpu="A10G",  # cheaper than A100, perfect for inference
    timeout=60 * 60,
    memory=32000,
    volumes={"/mnt/models": volume},
    image=modal.Image.debian_slim().pip_install([
        "torch>=2.1.0",
        "transformers>=4.34.0",
        "peft>=0.5.0",
        "accelerate",
        "sentencepiece"
    ]),
)
def run_batch_inference():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # -----------------------
    # Load tokenizer & model
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    # -----------------------
    # Read input prompts
    # -----------------------
    prompts = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split("|")]
            engagement = parts[0]
            topics = parts[1:]

            topic_tokens = " ".join(f"[{t}]" for t in topics)
            prompt = f"[engagement:{engagement}] {topic_tokens}\nIspovest:"

            prompts.append(prompt)

    print(f"Loaded {len(prompts)} prompts")

    # -----------------------
    # Generate
    # -----------------------
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        results.append({
            "input": prompt,
            "generated_text": text
        })

    # -----------------------
    # Save output
    # -----------------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} generations to {OUTPUT_FILE}")

# -----------------------
# LOCAL ENTRYPOINT
# -----------------------
@app.local_entrypoint()
def main():
    print("Starting batch inference...")
    app.deploy()
    run_batch_inference.remote()
