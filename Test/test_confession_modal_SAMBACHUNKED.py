import modal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -----------------------
# Modal app & volume
# -----------------------
app = modal.App("SambaLingo-confessions-testing")
volume = modal.Volume.from_name("my-volume")

promptPath = "prompts_samba_2.txt"

BASE_MODEL = "sambanovasystems/SambaLingo-Serbian-Chat"
ADAPTER_PATH = "/mnt/models/SambaLingo-confessions-20251223-145540"
PROMPTS_PATH = f"/mnt/models/{promptPath}"
OUTPUT_PATH = "/mnt/models/generated_confessions_SAMBA_2.txt"
OUTPUT_PATH_CHUNKED = "/mnt/models/generated_confessions_SAMBA_2_chunked.txt"

# -----------------------
# Common setup
# -----------------------
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model.eval()

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model, device

# -----------------------
# Original single-shot test
# -----------------------
@app.function(
    gpu="A100-40GB",
    timeout=60 * 60,
    memory=32000,
    volumes={"/mnt/models": volume},
    image=modal.Image.debian_slim().pip_install([
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "sentencepiece",
    ]),
)
def test_model():
    tokenizer, model, device = load_model_and_tokenizer()

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    outputs = []

    for idx, prompt in enumerate(prompts):
        prompt_input = prompt + "\n"
        inputs = tokenizer(prompt_input, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.12,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        continuation = text[len(prompt_input):].strip()
        outputs.append(f"### SAMPLE {idx+1}\n{prompt}\n{continuation}\n")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(outputs))

    print(f"✅ Generations written to {OUTPUT_PATH}")

# -----------------------
# Chunked generation for longer confessions
# -----------------------
@app.function(
    gpu="A100-40GB",
    timeout=60 * 60,
    memory=32000,
    volumes={"/mnt/models": volume},
    image=modal.Image.debian_slim().pip_install([
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "sentencepiece",
    ]),
)
def test_model_chunked():
    tokenizer, model, device = load_model_and_tokenizer()

    MAX_NEW_TOKENS_PER_CHUNK = 120
    TOTAL_TOKENS = 300
    TEMPERATURE = 0.85
    TOP_P = 0.9

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    outputs = []

    for idx, prompt in enumerate(prompts):
        prompt_input = prompt + "\n"
        current_text = prompt_input
        tokens_generated = 0

        while tokens_generated < TOTAL_TOKENS:
            inputs = tokenizer(current_text, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS_PER_CHUNK,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repetition_penalty=1.12,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            continuation = new_text[len(current_text):].strip()
            if not continuation:
                break  # stop if nothing new
            current_text += " " + continuation
            tokens_generated += len(tokenizer(continuation).input_ids)

        outputs.append(f"### SAMPLE {idx+1}\n{prompt}\n{current_text[len(prompt_input):]}\n")

    with open(OUTPUT_PATH_CHUNKED, "w", encoding="utf-8") as f:
        f.write("\n\n".join(outputs))

    print(f"✅ Chunked generations written to {OUTPUT_PATH_CHUNKED}")

# -----------------------
# Local entrypoint
# -----------------------
@app.local_entrypoint()
def main():
    print("🧪 Running model testing...")
    # Choose which function to run:
    # test_model.remote()
    test_model_chunked.remote()
