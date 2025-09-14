import json
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import os
from datetime import datetime
import warnings
import logging
from sklearn.model_selection import train_test_split
import numpy as np
import re

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HighQualityMotorcycleDataset(Dataset):
    """High-quality dataset with strict preprocessing for coherent outputs"""
    
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = self._preprocess_data()
        
    def _preprocess_data(self):
        """Strict preprocessing for better quality"""
        processed = []
        
        for item in self.data:
            # Clean and validate
            question = self._clean_text(item['question'])
            answer = self._clean_text(item['answer'])
            
            # Skip if not meeting quality standards
            if not self._is_high_quality(question, answer):
                continue
                
            processed.append({
                'question': question,
                'answer': answer,
                'category': item.get('category', 'general')
            })
        
        logger.info(f"Processed {len(processed)} high-quality entries from {len(self.data)} total")
        return processed
    
    def _clean_text(self, text):
        """Aggressive text cleaning for Indonesian motorcycle domain"""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-Indonesian characters and weird symbols
        text = re.sub(r'[^\w\s\.,\?\!\-\(\)\:]+', '', text)
        
        # Fix common Indonesian informal words
        replacements = {
            r'\b(ga|gak|gk)\b': 'tidak',
            r'\b(gimana|gmn)\b': 'bagaimana',
            r'\b(kenapa|knp)\b': 'mengapa',
            r'\b(udah|dah)\b': 'sudah',
            r'\b(aja|aj)\b': 'saja',
            r'\b(banget|bgt)\b': 'sangat',
            r'\b(terus|trs)\b': 'terus',
            r'\b(sama|sm)\b': 'dengan',
            r'\b(bisa|bs)\b': 'bisa',
            r'\b(motor gue|motor gw)\b': 'motor saya',
            r'\b(gue|gw|ane)\b': 'saya',
            r'\b(lu|lo|ente)\b': 'Anda'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def _is_high_quality(self, question, answer):
        """Strict quality check"""
        # Length requirements
        if len(question) < 15 or len(question) > 150:
            return False
        if len(answer) < 30 or len(answer) > 300:
            return False
        
        # Must contain Indonesian motorcycle-related content
        motorcycle_keywords = [
            'motor', 'mesin', 'oli', 'rem', 'ban', 'karbu', 'busi', 'shock',
            'rantai', 'gear', 'kopling', 'radiator', 'filter', 'bensin',
            'service', 'perawatan', 'perbaikan', 'ganti', 'cek', 'periksa'
        ]
        
        text_combined = (question + ' ' + answer).lower()
        if not any(keyword in text_combined for keyword in motorcycle_keywords):
            return False
        
        # Check for nonsensical patterns
        nonsense_patterns = [
            r'\b[a-zA-Z]{1,2}\s+[a-zA-Z]{1,2}\s+[a-zA-Z]{1,2}\b',  # Too many short words
            r'[a-zA-Z]{15,}',  # Very long words
            r'(.)\1{3,}',  # Repeated characters
            r'\d{8,}',  # Long numbers
            r'[A-Z]{5,}',  # Too many capitals
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, answer):
                return False
        
        return True
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        # Simple, clean format
        text = f"Pertanyaan: {item['question']}\nJawaban: {item['answer']}"
        
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
            'labels': encoding['input_ids'].flatten()
        }

def load_comprehensive_data(dataset_path):
    """Load the comprehensive dataset"""
    logger.info(f"Loading comprehensive dataset from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Split into train and validation
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    logger.info(f"Training set: {len(train_data)} entries")
    logger.info(f"Validation set: {len(val_data)} entries")
    
    return train_data, val_data

def setup_optimized_model_and_tokenizer(base_model_path):
    """Setup model with optimized configuration"""
    logger.info(f"Loading base model from {base_model_path}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(base_model_path)
    
    # Optimize model configuration for Indonesian
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"Model loaded with {model.num_parameters()} parameters")
    
    return model, tokenizer

def create_conservative_training_args(output_dir):
    """Conservative training arguments for stable, coherent output"""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,  # Fewer epochs to prevent overfitting
        per_device_train_batch_size=2,  # Very small batch
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size = 16
        warmup_steps=50,
        learning_rate=1e-5,  # Very conservative learning rate
        weight_decay=0.05,  # Higher weight decay
        logging_steps=20,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,  # Disable for stability
        lr_scheduler_type="linear",  # Simple linear scheduler
        adam_epsilon=1e-8,
        max_grad_norm=0.5,  # Aggressive gradient clipping
        report_to=[],
        run_name=f"final_enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

class StableTrainer(Trainer):
    """Stable trainer with conservative approach"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Conservative loss computation"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is not None:
            # Standard cross-entropy without label smoothing
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def train_final_enhanced_model(base_model_path, dataset_path):
    """Final enhanced training with strict quality control"""
    logger.info("Starting final enhanced training")
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"final_enhanced_motorcycle_model_{timestamp}"
    
    # Load data
    train_data, val_data = load_comprehensive_data(dataset_path)
    
    # Setup model
    model, tokenizer = setup_optimized_model_and_tokenizer(base_model_path)
    
    # Create high-quality datasets
    train_dataset = HighQualityMotorcycleDataset(train_data, tokenizer)
    val_dataset = HighQualityMotorcycleDataset(val_data, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = create_conservative_training_args(output_dir)
    
    # Create stable trainer with early stopping
    trainer = StableTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    logger.info("Starting conservative training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "timestamp": timestamp,
        "base_model_path": base_model_path,
        "final_model_path": output_dir,
        "dataset_path": dataset_path,
        "train_entries": len(train_dataset),
        "val_entries": len(val_dataset),
        "training_completed": True,
        "quality_improvements": [
            "Strict text preprocessing and cleaning",
            "High-quality dataset filtering",
            "Conservative training parameters",
            "Aggressive gradient clipping",
            "Early stopping for overfitting prevention",
            "Indonesian motorcycle domain focus",
            "Coherent output optimization"
        ]
    }
    
    with open(f"final_enhanced_training_info_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    return output_dir, training_info

def test_final_model(model_path):
    """Test the final model with comprehensive questions"""
    logger.info(f"Testing final model from {model_path}")
    
    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Comprehensive test questions
    test_questions = [
        "Motor saya susah hidup di pagi hari, apa penyebabnya?",
        "Bagaimana cara merawat rem cakram motor sport?",
        "Motor bergetar saat idle, solusinya bagaimana?",
        "Oli motor cepat habis, apa yang harus dicek?",
        "Cara setting karburator motor bebek yang benar?",
        "Motor tidak bisa distarter elektrik, mengapa?",
        "Tips agar motor irit bensin untuk touring?",
        "Shock motor bunyi, apakah berbahaya?",
        "Rantai motor kendor, bagaimana cara mengencangkan?",
        "Busi motor kotor, bisa dibersihkan sendiri?"
    ]
    
    model.eval()
    
    print("\n=== TESTING FINAL ENHANCED MODEL ===")
    for i, question in enumerate(test_questions, 1):
        prompt = f"Pertanyaan: {question}\nJawaban:"
        
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,  # Shorter responses
                num_return_sequences=1,
                temperature=0.6,  # Lower temperature for more coherent output
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,  # Higher repetition penalty
                no_repeat_ngram_size=3
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.replace(prompt, "").strip()
        
        print(f"\n{i}. {question}")
        print(f"Jawaban: {answer}")
        print("-" * 80)

def main():
    """Main function"""
    print("=== FINAL ENHANCED MOTORCYCLE AI TRAINING ===")
    
    # Configuration
    base_model_path = "fixed_motorcycle_model_20250914_090726"
    dataset_path = "comprehensive_motorcycle_dataset_20250914_095511.json"
    
    # Verify files exist
    if not os.path.exists(base_model_path):
        logger.error(f"Base model not found: {base_model_path}")
        return
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    try:
        # Train final model
        output_dir, training_info = train_final_enhanced_model(base_model_path, dataset_path)
        
        print(f"\n✅ FINAL ENHANCED TRAINING COMPLETED!")
        print(f"📍 Base Model: {base_model_path}")
        print(f"📍 Final Model: {output_dir}")
        print(f"📊 Dataset: {dataset_path}")
        print(f"📈 Training Entries: {training_info['train_entries']}")
        print(f"📈 Validation Entries: {training_info['val_entries']}")
        
        # Test the final model
        test_final_model(output_dir)
        
        print(f"\n🚀 Final enhanced motorcycle AI model is ready!")
        print(f"📁 Model saved to: {output_dir}")
        print(f"🎯 Optimized for coherent Indonesian motorcycle Q&A")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()