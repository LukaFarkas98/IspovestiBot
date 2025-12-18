from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "mistralai/Mistral-7B-v0.1"  # or whatever you used
adapter_path = "../training/mistral-confessions-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, adapter_path)

prompt = "reci mi kako si zavrsio u nemackoj:"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.9,
    top_p=0.95,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))