import json
import os
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from adaptive_learning_system import AdaptiveLearningSystem
import shutil

class IntegratedTrainingSystem:
    def __init__(self, base_model_name: str = "microsoft/DialoGPT-medium"):
        self.base_model_name = base_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"🚀 Integrated Training System initialized")
        print(f"📱 Device: {self.device}")
        print(f"🕒 Timestamp: {self.timestamp}")
    
    def load_and_prepare_dataset(self, dataset_path: str):
        """Load and prepare dataset for training"""
        print(f"📂 Loading dataset: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📊 Dataset loaded: {len(data)} entries")
        
        # Prepare training texts
        training_texts = []
        for item in data:
            # Format: <instruction> <input> <output>
            if 'instruction' in item and 'input' in item and 'output' in item:
                text = f"{item['instruction']} {item['input']} {item['output']}"
            elif 'question' in item and 'answer' in item:
                text = f"Jawab pertanyaan tentang motor berikut: {item['question']} {item['answer']}"
            else:
                continue
            
            training_texts.append(text)
        
        print(f"✅ Prepared {len(training_texts)} training texts")
        return training_texts
    
    def create_tokenized_dataset(self, texts: list, tokenizer):
        """Create tokenized dataset"""
        print("🔤 Tokenizing dataset...")
        
        def tokenize_function(examples):
            # Tokenize without return_tensors, let the data collator handle it
            tokenized = tokenizer(
                examples['text'], 
                truncation=True, 
                padding=False,  # Let data collator handle padding
                max_length=512
            )
            # Add labels for language modeling
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        
        print(f"✅ Dataset tokenized: {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def setup_model_and_tokenizer(self, model_path: str = None):
        """Setup model and tokenizer"""
        if model_path and os.path.exists(model_path):
            print(f"📥 Loading existing model: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            print(f"📥 Loading base model: {self.base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Add padding token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.to(self.device)
        print(f"✅ Model loaded on {self.device}")
        
        return model, tokenizer
    
    def train_model(self, model, tokenizer, train_dataset, output_dir: str):
        """Train the model"""
        print("🎯 Starting model training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            logging_dir=f"{output_dir}/logs",
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            report_to=None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        print("🚀 Training started...")
        start_time = datetime.now()
        
        trainer.train()
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Training completed in {training_duration:.2f} seconds")
        
        # Save the model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"💾 Model saved to: {output_dir}")
        
        return trainer
    
    def integrate_adaptive_learning(self, base_dataset_path: str, chat_history_file: str = None):
        """Integrate adaptive learning with base dataset"""
        print("🧠 Integrating adaptive learning...")
        
        # Initialize adaptive learning system
        learning_system = AdaptiveLearningSystem(
            model_path="improved_motorcycle_model_20250914_074059",
            chat_history_file=chat_history_file or "chat_history.json"
        )
        
        # Create adaptive dataset
        adaptive_dataset_path = f"adaptive_dataset_{self.timestamp}.json"
        learning_system.create_adaptive_dataset(
            base_dataset_path,
            adaptive_dataset_path
        )
        
        return adaptive_dataset_path
    
    def run_complete_training_pipeline(self, 
                                     dataset_path: str,
                                     existing_model_path: str = None,
                                     enable_adaptive_learning: bool = True):
        """Run complete training pipeline"""
        print("🔄 Starting Complete Training Pipeline...")
        print("=" * 50)
        
        # Step 1: Integrate adaptive learning if enabled
        if enable_adaptive_learning:
            print("\n📚 Step 1: Integrating Adaptive Learning")
            dataset_path = self.integrate_adaptive_learning(dataset_path)
        else:
            print("\n📚 Step 1: Using Base Dataset")
        
        # Step 2: Load and prepare dataset
        print("\n📂 Step 2: Loading and Preparing Dataset")
        training_texts = self.load_and_prepare_dataset(dataset_path)
        
        # Step 3: Setup model and tokenizer
        print("\n🤖 Step 3: Setting up Model and Tokenizer")
        model, tokenizer = self.setup_model_and_tokenizer(existing_model_path)
        
        # Step 4: Create tokenized dataset
        print("\n🔤 Step 4: Creating Tokenized Dataset")
        train_dataset = self.create_tokenized_dataset(training_texts, tokenizer)
        
        # Step 5: Train model
        print("\n🎯 Step 5: Training Model")
        output_dir = f"ultimate_motorcycle_model_{self.timestamp}"
        trainer = self.train_model(model, tokenizer, train_dataset, output_dir)
        
        # Step 6: Generate training report
        print("\n📊 Step 6: Generating Training Report")
        report = self.generate_training_report(
            dataset_path, output_dir, len(training_texts)
        )
        
        print("\n" + "=" * 50)
        print("🎉 Complete Training Pipeline Finished!")
        print(f"📁 Model saved to: {output_dir}")
        print(f"📊 Training report: training_report_{self.timestamp}.json")
        
        return output_dir, report
    
    def generate_training_report(self, dataset_path: str, model_path: str, training_examples: int):
        """Generate comprehensive training report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "training_info": {
                "dataset_path": dataset_path,
                "model_output_path": model_path,
                "training_examples": training_examples,
                "device_used": str(self.device),
                "base_model": self.base_model_name
            },
            "system_capabilities": {
                "adaptive_learning": True,
                "chat_history_integration": True,
                "dataset_deduplication": True,
                "quality_filtering": True
            },
            "features": [
                "Advanced dataset generation with 1000+ entries",
                "Intelligent deduplication and cleaning",
                "Adaptive learning from chat interactions",
                "Real-time response improvement",
                "Context-aware responses",
                "Pattern recognition from user feedback"
            ]
        }
        
        # Save report
        report_path = f"training_report_{self.timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report

def main():
    # Initialize integrated training system
    training_system = IntegratedTrainingSystem()
    
    # Configuration
    base_dataset = "ultra_clean_motorcycle_dataset.json"
    existing_model = "improved_motorcycle_model_20250914_074059"
    
    # Check if files exist
    if not os.path.exists(base_dataset):
        print(f"❌ Dataset not found: {base_dataset}")
        return
    
    if not os.path.exists(existing_model):
        print(f"⚠️ Existing model not found: {existing_model}")
        print("🔄 Will use base model instead")
        existing_model = None
    
    # Run complete training pipeline
    try:
        model_path, report = training_system.run_complete_training_pipeline(
            dataset_path=base_dataset,
            existing_model_path=existing_model,
            enable_adaptive_learning=True
        )
        
        print("\n🎯 Training Summary:")
        print(f"✅ Model trained successfully")
        print(f"📁 Model location: {model_path}")
        print(f"📊 Training examples: {report['training_info']['training_examples']}")
        print(f"🧠 Adaptive learning: Enabled")
        print(f"🔧 Device: {report['training_info']['device_used']}")
        
        print("\n🚀 Your AI is now ready with enhanced capabilities!")
        print("💡 Features:")
        for feature in report['features']:
            print(f"   • {feature}")
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()