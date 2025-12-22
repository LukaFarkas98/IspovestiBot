import modal

# -----------------------
# Modal app + volume
# -----------------------
app = modal.App("mistral-confessions-test")
volume = modal.Volume.from_name("mistral_models")

# -----------------------
# Reusable image
# -----------------------
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
    ])
)

# -----------------------
# Remote inference (SAFE)
# -----------------------
@app.function(
    gpu="A10G",
    timeout=300,
    volumes={"/mnt/models": volume},
    image=image,
)
def run_inference(prompts: list[str]):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    ADAPTER_DIR = "/mnt/models/mistral-confessions-20251220-181056"
    BASE_MODEL = "mistralai/Mistral-7B-v0.1"

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    # --- quant config (MATCH TRAINING) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # --- load base model ---
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    # --- attach adapter ---
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    # 🔥 HARD ASSERT: LoRA MUST be present
    assert hasattr(model, "peft_config"), "❌ LoRA adapter NOT loaded!"

    model.eval()

    # 🔍 Print proof once (shows LoRA modules)
    print(model)

    outputs = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        outputs.append(text)

    # --- save outputs ---
    out_path = "/mnt/models/test_outputs.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(outputs, 1):
            f.write(f"=== PROMPT {i} ===\n{text}\n\n")

    print(f"✅ Inference done. Saved to {out_path}")

# -----------------------
# Local entrypoint
# -----------------------
@app.local_entrypoint()
def main():
    with open("test_prompts.txt", "r", encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip()]

    run_inference.remote(prompts)
    print("🚀 Inference job submitted")
