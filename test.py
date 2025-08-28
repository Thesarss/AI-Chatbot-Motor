from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ===============================
# 1. Load Model & Tokenizer
# ===============================
model_path = "./bengkelAI-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Pastikan device pakai GPU jika tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===============================
# 2. Fungsi untuk Generate Jawaban
# ===============================
def chat(instruction, max_length=128):
    prompt = f"Instruksi: {instruction}\nJawaban:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # hanya ambil bagian setelah "Jawaban:"
    if "Jawaban:" in result:
        result = result.split("Jawaban:")[-1].strip()
    return result

# ===============================
# 3. Uji Coba
# ===============================
while True:
    instruksi = input("Masukkan instruksi (atau ketik 'exit' untuk keluar): ")
    if instruksi.lower() == "exit":
        break
    jawaban = chat(instruksi)
    print(f"🤖 {jawaban}\n")
