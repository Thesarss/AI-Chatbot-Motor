import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

# ===============================
# 1. Load Model + Tokenizer
# ===============================
base_model_name = "google/gemma-2b-it"
lora_adapter_path = "./bengkelAI-Gemma-LORA-final"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Device: {device}")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Gabungkan dengan LoRA
model = PeftModel.from_pretrained(base_model, lora_adapter_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===============================
# 2. Prompt Builder (Few-Shot)
# ===============================
def build_prompt(user_input):
    contoh = (
        "Anda adalah asisten bengkel AI. Jawab keluhan motor "
        "dengan format KETAT (wajib 3 baris, tanpa tambahan lain):\n\n"
        "Permasalahan: ...\n"
        "Solusi: ...\n"
        "Estimasi Biaya: ...\n\n"
        "Contoh:\n"
        "Keluhan: motor susah hidup\n"
        "Jawaban:\n"
        "Permasalahan: Aki lemah atau busi kotor\n"
        "Solusi: Periksa aki dan bersihkan/ganti busi\n"
        "Estimasi Biaya: Rp100.000 - Rp300.000\n\n"
        f"Keluhan: {user_input}\n"
        "Jawaban:"
    )
    chat = [{"role": "user", "content": contoh}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# ===============================
# 3. Output Parser
# ===============================
def parse_output(output):
    hasil = {"Permasalahan": "-", "Solusi": "-", "Estimasi Biaya": "-"}

    # Regex aman
    permasalahan = re.search(r"Permasalahan:\s*(.*?)(Solusi:|Estimasi Biaya:|$)", output, re.S)
    solusi = re.search(r"Solusi:\s*(.*?)(Estimasi Biaya:|$)", output, re.S)
    biaya = re.search(r"Estimasi Biaya:\s*(.*)", output, re.S)

    if permasalahan:
        hasil["Permasalahan"] = permasalahan.group(1).strip()
    if solusi:
        hasil["Solusi"] = solusi.group(1).strip()
    if biaya:
        hasil["Estimasi Biaya"] = biaya.group(1).strip()

    # Fallback kalau kosong
    if hasil["Permasalahan"] == "-":
        hasil["Permasalahan"] = "Kemungkinan ada kerusakan pada komponen terkait"
    if hasil["Solusi"] == "-":
        hasil["Solusi"] = "Periksa komponen terkait atau bawa ke bengkel terdekat"
    if hasil["Estimasi Biaya"] == "-":
        hasil["Estimasi Biaya"] = "Rp100.000 - Rp500.000 (estimasi umum)"

    return hasil

# ===============================
# 4. Chat Loop
# ===============================
print("\n🤖 BengkelAI siap membantu! (ketik 'exit' untuk keluar)\n")

while True:
    user_input = input("Anda: ")
    if user_input.lower().strip() == "exit":
        print("👋 Terima kasih!")
        break

    prompt = build_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,  # cukup untuk 3 baris
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Jawaban:")[-1].strip()

    # 🔹 Stop parsing setelah "Estimasi Biaya:" pertama
    if "Estimasi Biaya:" in answer:
        answer = answer.split("Estimasi Biaya:")[0] + \
                 "Estimasi Biaya: " + answer.split("Estimasi Biaya:")[1].split("\n")[0]

    hasil = parse_output(answer)

    print("\n🤖 BengkelAI:")
    print(f"Permasalahan: {hasil['Permasalahan']}")
    print(f"Solusi: {hasil['Solusi']}")
    print(f"Estimasi Biaya: {hasil['Estimasi Biaya']}")
    print("-" * 40)
