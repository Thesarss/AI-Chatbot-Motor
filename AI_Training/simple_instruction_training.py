import json
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import os
from datetime import datetime
import warnings
import logging
from sklearn.model_selection import train_test_split
import re

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleInstructionDataset(Dataset):
    """Simple instruction-following dataset with minimal preprocessing"""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = self._create_simple_instructions(data)
        
    def _create_simple_instructions(self, data):
        """Create simple, clean instruction-response pairs"""
        processed = []
        
        for item in data:
            question = self._clean_simple(item.get('question', ''))
            answer = self._clean_simple(item.get('answer', ''))
            
            # Skip if too short or too long
            if len(question) < 10 or len(question) > 100:
                continue
            if len(answer) < 20 or len(answer) > 150:
                continue
            
            # Must contain motorcycle keywords
            if not self._contains_motorcycle_content(question + ' ' + answer):
                continue
            
            # Simple format
            instruction = f"Jawab pertanyaan tentang motor: {question}"
            response = answer
            
            processed.append({
                'instruction': instruction,
                'response': response
            })
        
        logger.info(f"Created {len(processed)} simple instruction pairs from {len(data)} entries")
        return processed
    
    def _clean_simple(self, text):
        """Very simple text cleaning"""
        if not text:
            return ""
        
        # Basic cleaning only
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\.,\?\!\-\(\)\:]', '', text)
        
        # Fix common informal words
        text = re.sub(r'\b(ga|gak)\b', 'tidak', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(gimana)\b', 'bagaimana', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(kenapa)\b', 'mengapa', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(udah)\b', 'sudah', text, flags=re.IGNORECASE)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def _contains_motorcycle_content(self, text):
        """Check if text contains motorcycle-related content"""
        keywords = [
            'motor', 'mesin', 'oli', 'rem', 'ban', 'karbu', 'busi',
            'shock', 'rantai', 'gear', 'kopling', 'bensin', 'service'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        # Very simple format
        text = f"{item['instruction']}\n{item['response']}<|endoftext|>"
        
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

def load_and_filter_data(dataset_path):
    """Load and filter data for simple instruction training"""
    logger.info(f"Loading data from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take only a subset for focused training
    if len(data) > 400:
        data = data[:400]  # Limit to 400 best entries
    
    # Split
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)
    
    logger.info(f"Using {len(train_data)} training and {len(val_data)} validation entries")
    return train_data, val_data

def setup_simple_model(base_model_path):
    """Setup model for simple instruction following"""
    logger.info(f"Loading model from {base_model_path}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)
    model = GPT2LMHeadModel.from_pretrained(base_model_path)
    
    # Add special tokens
    special_tokens = {'pad_token': '<|pad|>', 'eos_token': '<|endoftext|>'}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def create_minimal_training_args(output_dir):
    """Minimal training arguments for stable learning"""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,  # Very few epochs
        per_device_train_batch_size=1,  # Smallest batch
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Small effective batch
        warmup_steps=20,
        learning_rate=5e-6,  # Very small learning rate
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=False,  # Disable for stability
        gradient_checkpointing=False,
        lr_scheduler_type="constant",  # Constant learning rate
        max_grad_norm=0.3,  # Very aggressive clipping
        report_to=[],
        run_name=f"simple_instruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

def train_simple_instruction_model(base_model_path, dataset_path):
    """Train simple instruction-following model"""
    logger.info("Starting simple instruction training")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"simple_instruction_motorcycle_model_{timestamp}"
    
    # Load data
    train_data, val_data = load_and_filter_data(dataset_path)
    
    # Setup model
    model, tokenizer = setup_simple_model(base_model_path)
    
    # Create datasets
    train_dataset = SimpleInstructionDataset(train_data, tokenizer)
    val_dataset = SimpleInstructionDataset(val_data, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = create_minimal_training_args(output_dir)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    logger.info("Starting minimal training...")
    trainer.train()
    
    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

def test_simple_model(model_path):
    """Test the simple instruction model"""
    logger.info(f"Testing simple model from {model_path}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Simple test questions
    test_questions = [
        "Motor susah hidup pagi hari kenapa?",
        "Cara merawat rem motor?",
        "Motor bergetar saat idle?",
        "Oli motor cepat habis?",
        "Cara setting karburator?"
    ]
    
    model.eval()
    
    print("\n=== TESTING SIMPLE INSTRUCTION MODEL ===")
    for i, question in enumerate(test_questions, 1):
        prompt = f"Jawab pertanyaan tentang motor: {question}\n"
        
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,  # Short responses
                num_return_sequences=1,
                temperature=0.3,  # Very low temperature
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.5,
                no_repeat_ngram_size=2
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.replace(prompt, "").strip()
        
        print(f"\n{i}. {question}")
        print(f"Jawaban: {answer}")
        print("-" * 60)

def main():
    """Main function for simple instruction training"""
    print("=== SIMPLE INSTRUCTION MOTORCYCLE AI TRAINING ===")
    
    base_model_path = "fixed_motorcycle_model_20250914_090726"
    dataset_path = "comprehensive_motorcycle_dataset_20250914_095511.json"
    
    # Verify files
    if not os.path.exists(base_model_path):
        logger.error(f"Base model not found: {base_model_path}")
        return
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    try:
        # Train
        output_dir = train_simple_instruction_model(base_model_path, dataset_path)
        
        print(f"\n✅ SIMPLE INSTRUCTION TRAINING COMPLETED!")
        print(f"📍 Model saved to: {output_dir}")
        
        # Test
        test_simple_model(output_dir)
        
        print(f"\n🎯 Simple instruction motorcycle model ready!")
        print(f"📁 Location: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()