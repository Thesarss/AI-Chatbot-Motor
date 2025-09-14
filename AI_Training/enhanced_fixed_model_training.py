import json
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import os
from datetime import datetime
import warnings
import logging
from sklearn.model_selection import train_test_split
import numpy as np

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMotorcycleDataset(Dataset):
    """Enhanced dataset class for motorcycle Q&A with better preprocessing"""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Enhanced format with context
        if 'category' in item and 'subcategory' in item:
            text = f"Kategori: {item['category']} - {item['subcategory']}\nPertanyaan: {item['question']}\nJawaban: {item['answer']}"
        else:
            text = f"Pertanyaan: {item['question']}\nJawaban: {item['answer']}"
        
        # Tokenize with proper attention
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

def load_and_prepare_data(dataset_path):
    """Load and prepare the enhanced dataset"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter out invalid entries
    valid_data = []
    for item in data:
        if ('question' in item and 'answer' in item and 
            len(item['question'].strip()) > 5 and 
            len(item['answer'].strip()) > 10):
            valid_data.append(item)
    
    logger.info(f"Loaded {len(valid_data)} valid entries from {len(data)} total entries")
    
    # Split into train and validation
    train_data, val_data = train_test_split(valid_data, test_size=0.15, random_state=42)
    
    logger.info(f"Training set: {len(train_data)} entries")
    logger.info(f"Validation set: {len(val_data)} entries")
    
    return train_data, val_data

def setup_enhanced_model_and_tokenizer(base_model_path):
    """Setup model and tokenizer from existing trained model"""
    logger.info(f"Loading base model from {base_model_path}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load pre-trained model
    model = GPT2LMHeadModel.from_pretrained(base_model_path)
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"Model loaded successfully with {model.num_parameters()} parameters")
    
    return model, tokenizer

def create_enhanced_training_args(output_dir):
    """Create optimized training arguments for continued training"""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,  # More epochs for better learning
        per_device_train_batch_size=3,  # Smaller batch for stability
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 12
        warmup_steps=200,
        learning_rate=2e-5,  # Lower learning rate for fine-tuning
        weight_decay=0.01,
        logging_steps=25,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        lr_scheduler_type="cosine_with_restarts",
        adam_epsilon=1e-6,
        max_grad_norm=1.0,
        report_to=[],  # Disable wandb
        run_name=f"enhanced_motorcycle_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

class EnhancedTrainer(Trainer):
    """Enhanced trainer with custom loss computation"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation with label smoothing"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        if labels is not None:
            # Apply label smoothing
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def train_enhanced_model(base_model_path, dataset_path):
    """Main training function for enhanced model"""
    logger.info("Starting enhanced model training")
    
    # Setup timestamp and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"enhanced_fixed_motorcycle_model_{timestamp}"
    
    # Load data
    train_data, val_data = load_and_prepare_data(dataset_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_enhanced_model_and_tokenizer(base_model_path)
    
    # Create datasets
    train_dataset = EnhancedMotorcycleDataset(train_data, tokenizer)
    val_dataset = EnhancedMotorcycleDataset(val_data, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = create_enhanced_training_args(output_dir)
    
    # Create enhanced trainer
    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving enhanced model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "timestamp": timestamp,
        "base_model_path": base_model_path,
        "enhanced_model_path": output_dir,
        "dataset_path": dataset_path,
        "train_entries": len(train_data),
        "val_entries": len(val_data),
        "total_entries": len(train_data) + len(val_data),
        "training_completed": True,
        "enhancements_applied": [
            "Continued training from fixed_motorcycle_model_20250914_090726",
            "Enhanced dataset with 4890+ entries",
            "Improved preprocessing with category context",
            "Label smoothing for better generalization",
            "Cosine learning rate scheduler with restarts",
            "Gradient checkpointing for memory efficiency",
            "Custom loss computation"
        ]
    }
    
    with open(f"enhanced_training_info_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    return output_dir, training_info

def test_enhanced_model(model_path):
    """Test the enhanced model with sample questions"""
    logger.info(f"Testing enhanced model from {model_path}")
    
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test questions
    test_questions = [
        "Motor saya susah hidup di pagi hari, apa penyebabnya?",
        "Bagaimana cara merawat rem cakram motor sport?",
        "Motor bergetar saat idle, solusinya gimana?",
        "Oli motor cepat habis, apa yang harus dicek?",
        "Cara setting karburator motor bebek yang benar?",
        "Motor tidak bisa distarter elektrik, kenapa ya?",
        "Tips agar motor irit bensin untuk touring?"
    ]
    
    model.eval()
    
    print("\n=== TESTING ENHANCED MODEL ===")
    for i, question in enumerate(test_questions, 1):
        prompt = f"Pertanyaan: {question}\nJawaban:"
        
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.replace(prompt, "").strip()
        
        print(f"\n{i}. {question}")
        print(f"Jawaban: {answer}")
        print("-" * 80)

def main():
    """Main function"""
    print("=== ENHANCED FIXED MODEL TRAINING ===")
    
    # Configuration
    base_model_path = "fixed_motorcycle_model_20250914_090726"
    dataset_path = "adaptive_motorcycle_dataset.json"
    
    # Check if base model exists
    if not os.path.exists(base_model_path):
        logger.error(f"Base model not found: {base_model_path}")
        return
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    # Train enhanced model
    try:
        output_dir, training_info = train_enhanced_model(base_model_path, dataset_path)
        
        print(f"\n✅ ENHANCED TRAINING COMPLETED!")
        print(f"📍 Base Model: {base_model_path}")
        print(f"📍 Enhanced Model: {output_dir}")
        print(f"📊 Dataset: {dataset_path}")
        print(f"📈 Training Entries: {training_info['train_entries']}")
        print(f"📈 Validation Entries: {training_info['val_entries']}")
        print(f"📈 Total Entries: {training_info['total_entries']}")
        
        # Test the enhanced model
        test_enhanced_model(output_dir)
        
        print(f"\n🚀 Enhanced motorcycle AI model is ready!")
        print(f"📁 Model saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()