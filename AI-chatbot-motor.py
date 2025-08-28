from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# 1. Load base model (Gemma 2B, lebih ringan dari LLaMA)
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# 2. Load dataset
dataset = load_dataset("json", data_files="otomotif_dataset.json")

# 3. Preprocessing
def tokenize(batch):
    return tokenizer(batch["instruction"], text_target=batch["response"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)

# 4. Training setup
training_args = TrainingArguments(
    output_dir="./chatbot-otomotif",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
)

# 5. Train model
trainer.train()

# 6. Save model
model.save_pretrained("./chatbot-otomotif")
tokenizer.save_pretrained("./chatbot-otomotif")
