import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
import re

class IndonesianTextPreprocessor:
    """Preprocessor khusus untuk teks bahasa Indonesia"""
    
    def __init__(self):
        # Kamus normalisasi kata-kata umum dalam konteks motor
        self.word_normalization = {
            'gimana': 'bagaimana',
            'kenapa': 'mengapa',
            'kira2': 'kira-kira',
            'udah': 'sudah',
            'udh': 'sudah',
            'blm': 'belum',
            'trus': 'terus',
            'yg': 'yang',
            'dgn': 'dengan',
            'utk': 'untuk',
            'krn': 'karena',
            'hrs': 'harus',
            'jd': 'jadi',
            'lg': 'lagi',
            'aja': 'saja',
            'bgt': 'banget',
            'bisa': 'dapat',
            'gak': 'tidak',
            'ga': 'tidak',
            'nggak': 'tidak',
            'ngga': 'tidak'
        }
        
        # Pattern untuk membersihkan teks
        self.cleaning_patterns = [
            (r'\s+', ' '),  # Multiple spaces
            (r'[^\w\s\?\!\.,]', ''),  # Non-alphanumeric except punctuation
            (r'\b(\w)\1{2,}\b', r'\1\1'),  # Repeated characters
        ]
    
    def normalize_text(self, text):
        """Normalisasi teks bahasa Indonesia"""
        # Convert to lowercase
        text = text.lower()
        
        # Normalize common words
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Remove punctuation for normalization check
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.word_normalization:
                # Replace with normalized version, keeping punctuation
                normalized_word = word.replace(clean_word, self.word_normalization[clean_word])
                normalized_words.append(normalized_word)
            else:
                normalized_words.append(word)
        
        text = ' '.join(normalized_words)
        
        # Apply cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Capitalize first letter
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def validate_text_quality(self, text):
        """Validasi kualitas teks"""
        # Cek panjang minimum
        if len(text.split()) < 3:
            return False
        
        # Cek apakah ada kata yang masuk akal
        motor_keywords = [
            'motor', 'mesin', 'oli', 'rem', 'ban', 'rantai', 'karburator',
            'busi', 'filter', 'kopling', 'transmisi', 'suspensi', 'shock',
            'knalpot', 'radiator', 'aki', 'lampu', 'spion', 'velg'
        ]
        
        text_lower = text.lower()
        has_motor_context = any(keyword in text_lower for keyword in motor_keywords)
        
        # Cek apakah teks mengandung kata-kata aneh
        weird_patterns = [
            r'\b\w{1,2}\b.*\b\w{1,2}\b.*\b\w{1,2}\b',  # Too many short words
            r'\b(\w+)\s+\1\s+\1\b',  # Repeated words
            r'[a-z]{10,}',  # Very long words without spaces
        ]
        
        for pattern in weird_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return has_motor_context

class OptimizedIndonesianDataset(Dataset):
    def __init__(self, data, tokenizer, preprocessor, max_length=256):
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_length = max_length
        
        # Preprocess dan filter data
        self.processed_data = []
        for item in data:
            question = self.preprocessor.normalize_text(item['question'])
            answer = self.preprocessor.normalize_text(item['answer'])
            
            # Validasi kualitas
            if (self.preprocessor.validate_text_quality(question) and 
                self.preprocessor.validate_text_quality(answer)):
                self.processed_data.append({
                    'question': question,
                    'answer': answer
                })
        
        print(f"Dataset difilter: {len(data)} -> {len(self.processed_data)} entri")
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        # Format yang lebih natural untuk bahasa Indonesia
        text = f"Q: {item['question']}\nA: {item['answer']}{self.tokenizer.eos_token}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten().clone()
        }

def setup_indonesian_model():
    """Setup model yang lebih cocok untuk bahasa Indonesia"""
    # Gunakan model yang lebih kecil dan stabil
    model_name = "microsoft/DialoGPT-small"
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer dan model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Setup pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def create_conservative_training_args(output_dir):
    """Training arguments yang lebih konservatif"""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters - sangat konservatif
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        
        # Learning rate yang sangat kecil
        learning_rate=1e-5,
        warmup_steps=20,
        
        # Strong regularization
        weight_decay=0.1,
        
        # Frequent evaluation
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=5,
        save_steps=20,
        
        # Early stopping
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Memory dan gradient optimization
        gradient_checkpointing=True,
        max_grad_norm=0.5,  # Sangat konservatif
        
        # Reproducibility
        seed=42,
        data_seed=42,
        
        # Disable logging
        report_to=[],
        
        # Save strategy
        save_total_limit=1,
        
        # Additional stability
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

class StableTrainer(Trainer):
    """Trainer dengan stabilitas ekstra"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 5
    
    def log(self, logs):
        super().log(logs)
        
        # Monitor untuk early stopping manual
        if 'eval_loss' in logs:
            eval_loss = logs['eval_loss']
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            print(f"Eval Loss: {eval_loss:.4f} | Best: {self.best_loss:.4f} | Patience: {self.patience_counter}/{self.max_patience}")
        
        if 'loss' in logs:
            step = logs.get('step', 0)
            epoch = logs.get('epoch', 0)
            loss = logs.get('loss', 0)
            lr = logs.get('learning_rate', 0)
            
            if step % 5 == 0:
                print(f"Step {step} | Epoch {epoch:.2f} | Loss: {loss:.4f} | LR: {lr:.2e}")

def train_indonesian_optimized_model():
    """Training dengan optimasi khusus bahasa Indonesia"""
    print("=== TRAINING MODEL OPTIMIZED UNTUK BAHASA INDONESIA ===")
    
    # Load dataset
    files = [f for f in os.listdir('.') if f.startswith('super_clean_motorcycle_dataset_')]
    if not files:
        raise FileNotFoundError("Dataset super clean tidak ditemukan!")
    
    latest_file = sorted(files)[-1]
    print(f"Menggunakan dataset: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Setup preprocessor
    preprocessor = IndonesianTextPreprocessor()
    
    # Setup model
    model, tokenizer = setup_indonesian_model()
    
    # Split data
    train_data, eval_data = train_test_split(data, test_size=0.3, random_state=42)
    
    print(f"Data training: {len(train_data)} entri")
    print(f"Data evaluasi: {len(eval_data)} entri")
    
    # Create datasets dengan preprocessing
    train_dataset = OptimizedIndonesianDataset(train_data, tokenizer, preprocessor)
    eval_dataset = OptimizedIndonesianDataset(eval_data, tokenizer, preprocessor)
    
    # Setup training
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"indonesian_optimized_model_{timestamp}"
    
    training_args = create_conservative_training_args(output_dir)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = StableTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("\nMemulai training dengan pendekatan konservatif...")
    
    # Training
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Training info
    training_info = {
        'timestamp': timestamp,
        'model_path': output_dir,
        'dataset_file': latest_file,
        'training_entries': len(train_dataset),
        'eval_entries': len(eval_dataset),
        'preprocessing_applied': True,
        'training_completed': True,
        'best_eval_loss': trainer.best_loss
    }
    
    info_file = f"indonesian_training_info_{timestamp}.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== TRAINING SELESAI ===")
    print(f"Model disimpan ke: {output_dir}")
    print(f"Best eval loss: {trainer.best_loss:.4f}")
    
    return output_dir, training_info

def test_indonesian_model(model_path):
    """Test model dengan pertanyaan bahasa Indonesia"""
    print(f"\n=== TESTING MODEL: {model_path} ===")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test questions
    test_questions = [
        "Motor saya susah hidup, apa penyebabnya?",
        "Bagaimana cara merawat rem motor?",
        "Motor bergetar saat idle, mengapa?",
        "Oli motor cepat habis, apa masalahnya?",
        "Tips agar motor irit bensin?",
        "Suara mesin kasar, apa penyebabnya?",
        "Rem motor blong, bagaimana mengatasinya?"
    ]
    
    print("\nTesting kualitas model:")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}/{len(test_questions)}]")
        print(f"❓ Q: {question}")
        
        # Generate dengan parameter yang lebih konservatif
        input_text = f"Q: {question}\nA:"
        inputs = tokenizer.encode(input_text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 80,
                num_return_sequences=1,
                temperature=0.6,  # Lebih konservatif
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )
        
        # Decode
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.replace(input_text, "").strip()
        
        print(f"🤖 A: {answer}")
        print("-" * 60)

def main():
    try:
        # Training
        model_path, training_info = train_indonesian_optimized_model()
        
        # Testing
        test_indonesian_model(model_path)
        
        print(f"\n✅ Training dan testing selesai!")
        print(f"Model tersimpan di: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()