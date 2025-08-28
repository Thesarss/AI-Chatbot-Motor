import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# ===============================
# 1. Load Dataset
# ===============================
dataset = load_dataset("json", data_files="otomotif_dataset.json")

# ===============================
# 2. Load Tokenizer & Model
# ===============================
model_name = "gpt2"   # bisa diganti dengan model lain misalnya "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# FIX pad_token agar tidak error
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# ===============================
# 3. Tokenisasi Dataset
# ===============================
def tokenize_function(example):
    # gabungkan category + instruction + response sebagai satu text
    text = f"Kategori: {example.get('category', 'Umum')}\nInstruksi: {example['instruction']}\nJawaban: {example['response']}"
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,   # lebih panjang biar tidak kepotong
    )
    tokens["labels"] = tokens["input_ids"].copy()  # penting supaya ada target
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=False)

# ===============================
# 4. Training Arguments
# ===============================
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",   # diperbaiki dari eval_strategy
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

# ===============================
# 5. Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"],  # sementara masih pakai train juga
)

# ===============================
# 6. Mulai Training
# ===============================
trainer.train()

# ===============================
# 7. Save Model
# ===============================
model.save_pretrained("./bengkelAI-model")
tokenizer.save_pretrained("./bengkelAI-model")

print("✅ Training selesai, model tersimpan di ./bengkelAI-model")
