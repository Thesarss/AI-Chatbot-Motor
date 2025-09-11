import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import notebook_login, login

# ===================================================================================
# CATATAN PENTING:
# Untuk menjalankan skrip ini, Anda perlu login ke Hugging Face terlebih dahulu.
#
# Opsi 1: Jika di Google Colab atau Jupyter Notebook, jalankan baris ini di sel terpisah:
#   notebook_login() 
#   Lalu masukkan token Anda.
#
# Opsi 2: Jika di terminal lokal, jalankan perintah ini di terminal Anda:
#   huggingface-cli login
#   Lalu masukkan token Anda.
# ===================================================================================


# ===============================
# 1. Konfigurasi
# ===============================
# Nama model dasar dari Hugging Face
base_model_name = "google/gemma-2b-it"
# Nama file dataset Anda
dataset_file = "otomotif_dataset_clean.json"
# Nama folder untuk menyimpan hasil training (adapter LoRA)
output_dir = "./bengkelAI-Gemma-LORA-final"

# ===============================
# 2. Load Dataset
# ===============================
print(f"Memuat dataset dari {dataset_file}...")
dataset = load_dataset("json", data_files=dataset_file, split="train")

# ===============================
# 3. Load Tokenizer & Model (Gemma 2B-it)
# ===============================
print(f"Memuat base model: {base_model_name}...")

# Konfigurasi Kuantisasi 4-bit untuk menghemat memori
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Muat model dasar dengan kuantisasi
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto" # Otomatis menggunakan GPU jika tersedia
)
model.config.use_cache = False # Penting untuk training LoRA

# Muat tokenizer yang sesuai dengan model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token # Gemma menggunakan end-of-sentence token sebagai padding

# ===============================
# 4. Konfigurasi LoRA
# ===============================
print("Mengkonfigurasi adapter LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # Target module untuk Gemma 2B
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Terapkan adapter LoRA ke model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===============================
# 5. Format Dataset Sesuai Template Chat Gemma
# ===============================
print("Memformat dataset dengan chat template...")
def format_prompt_gemma(example):
    # Membuat format chat yang akan dipelajari oleh model
    chat = [
        {"role": "user", "content": f"[Kategori: {example['category']}] Keluhan: {example['instruction']}"},
        {"role": "model", "content": f"Permasalahan: {example['response']['permasalahan']}\nSolusi: {example['response']['solusi']}\nEstimasi Biaya: {example['response']['biaya']}"}
    ]
    # Terapkan template dari tokenizer
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return {"text": prompt}

formatted_dataset = dataset.map(format_prompt_gemma)
tokenized_datasets = formatted_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, max_length=512), batched=True)

# Split dataset menjadi data training dan evaluasi
split_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Data collator untuk menangani batching dan padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ===============================
# 6. Training Arguments & Trainer
# ===============================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5, # Jumlah epoch bisa disesuaikan (3-5 adalah awal yang baik)
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True, # Gunakan mixed precision jika GPU mendukung (T4 di Colab mendukung)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ===============================
# 7. Mulai Training
# ===============================
print("Memulai proses training...")
trainer.train()

# ===============================
# 8. Simpan Model Final
# ===============================
print(f"✅ Training selesai. Menyimpan model final di {output_dir}...")
trainer.save_model(output_dir)
print("Model berhasil disimpan.")