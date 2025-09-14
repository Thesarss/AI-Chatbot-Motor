import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
import os
from typing import List, Dict, Tuple
import uuid
from chat_history_manager import ChatHistoryManager
from adaptive_learning_system import AdaptiveLearningSystem
from slang_preprocessor import IndonesianSlangPreprocessor
from flexible_similarity_search import FlexibleSimilaritySearch

class MotorcycleQADataset(Dataset):
    """Dataset class untuk motorcycle Q&A dengan preprocessing optimal"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess data untuk training yang optimal"""
        processed = []
        
        for item in self.data:
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            
            if not question or not answer:
                continue
                
            # Format input untuk training
            # Menggunakan format khusus untuk motorcycle Q&A
            input_text = f"<|startoftext|>Pertanyaan: {question}\nJawaban: {answer}<|endoftext|>"
            
            # Tokenize dengan padding dan truncation
            encoding = self.tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            processed.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': encoding['input_ids'].squeeze().clone()
            })
            
        return processed
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

class OptimizedMotorcycleAI:
    """Sistem AI Motor yang dioptimalkan untuk dataset new_motorcycle_qa_dataset.json"""
    
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model dan tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token jika belum ada
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Add special tokens untuk motorcycle domain
        special_tokens = {
            "additional_special_tokens": [
                "<|startoftext|>", "<|endoftext|>",
                "<|question|>", "<|answer|>",
                "<|motorcycle|>", "<|problem|>", "<|solution|>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        
        # Initialize adaptive learning components
        self.session_id = str(uuid.uuid4())
        self.chat_manager = ChatHistoryManager()
        self.adaptive_system = AdaptiveLearningSystem()
        self.chat_manager.create_session(self.session_id)
        
        # Initialize new components untuk bahasa gaul dan flexible search
        self.slang_preprocessor = IndonesianSlangPreprocessor()
        self.flexible_search = FlexibleSimilaritySearch(
            similarity_threshold=0.3,
            max_results=5,
            use_svd=True,
            svd_components=100
        )
        
        # Dataset dan similarity search
        self.dataset = None
        self.raw_data = []
        self.vectorizer = TfidfVectorizer(
            stop_words=None,  # Tidak menggunakan stop words untuk bahasa Indonesia
            ngram_range=(1, 3),  # Unigram, bigram, trigram
            max_features=5000,
            lowercase=True
        )
        self.question_vectors = None
        
        # Training parameters yang dioptimalkan
        self.training_args = TrainingArguments(
            output_dir='./motorcycle_ai_checkpoints',
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=2,  # Disesuaikan untuk GPU
            per_device_eval_batch_size=2,
            warmup_steps=100,
            logging_steps=50,
            save_steps=200,
            eval_steps=200,
            evaluation_strategy="steps",
            save_total_limit=3,
            prediction_loss_only=True,
            learning_rate=5e-5,
            weight_decay=0.01,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            fp16=torch.cuda.is_available(),  # Mixed precision untuk efisiensi
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        print(f"🚀 OptimizedMotorcycleAI initialized on {self.device}")
        print(f"📊 Model: {model_name}")
        print(f"🔧 Vocabulary size: {len(self.tokenizer)}")
    
    def load_dataset(self, dataset_path: str):
        """Load dan preprocess dataset dengan optimasi dan flexible search"""
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            
            print(f"📁 Dataset loaded: {len(self.raw_data)} entries")
            
            # Validasi dan cleaning data
            valid_data = []
            for i, item in enumerate(self.raw_data):
                if isinstance(item, dict) and 'question' in item and 'answer' in item:
                    question = str(item['question']).strip()
                    answer = str(item['answer']).strip()
                    
                    if len(question) > 10 and len(answer) > 10:  # Filter data yang terlalu pendek
                        valid_data.append({
                            'question': question,
                            'answer': answer
                        })
                else:
                    print(f"⚠️ Skipping invalid entry at index {i}")
            
            self.raw_data = valid_data
            print(f"✅ Valid data: {len(self.raw_data)} entries")
            
            # Load multiple datasets untuk flexible search
            self._load_multiple_datasets(dataset_path)
            
            # Buat similarity search vectors (legacy)
            self._build_similarity_search()
            
            # Initialize flexible similarity search
            self._initialize_flexible_search(dataset_path)
            
            # Create dataset untuk training
            self.dataset = MotorcycleQADataset(
                self.raw_data, 
                self.tokenizer, 
                max_length=512
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def _load_multiple_datasets(self, primary_dataset_path: str):
        """Load multiple datasets untuk comprehensive search"""
        dataset_files = [
            primary_dataset_path,
            "enhanced_motorcycle_dataset_400.json"
        ]
        
        all_data = []
        for dataset_file in dataset_files:
            try:
                if os.path.exists(dataset_file):
                    with open(dataset_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_data.extend(data)
                        print(f"📁 Loaded additional dataset: {dataset_file} ({len(data)} entries)")
            except Exception as e:
                print(f"⚠️ Could not load {dataset_file}: {e}")
        
        # Remove duplicates berdasarkan question similarity
        unique_data = []
        seen_questions = set()
        
        for item in all_data:
            question_normalized = self.slang_preprocessor.normalize_text(item.get('question', ''))
            if question_normalized not in seen_questions:
                unique_data.append(item)
                seen_questions.add(question_normalized)
        
        self.comprehensive_data = unique_data
        print(f"📊 Total comprehensive dataset: {len(self.comprehensive_data)} unique entries")
    
    def _initialize_flexible_search(self, dataset_path: str):
        """Initialize flexible similarity search dengan comprehensive dataset"""
        try:
            # Create temporary comprehensive dataset file
            temp_dataset_path = "temp_comprehensive_dataset.json"
            with open(temp_dataset_path, 'w', encoding='utf-8') as f:
                json.dump(self.comprehensive_data, f, ensure_ascii=False, indent=2)
            
            # Load ke flexible search
            self.flexible_search.load_dataset(temp_dataset_path)
            
            # Clean up temporary file
            if os.path.exists(temp_dataset_path):
                os.remove(temp_dataset_path)
            
            print("🔍 Flexible similarity search initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Error initializing flexible search: {e}")
    
    def _build_similarity_search(self):
        """Build TF-IDF vectors untuk similarity search (legacy)"""
        if not self.raw_data:
            return
            
        questions = [item['question'] for item in self.raw_data]
        
        # Preprocessing text untuk bahasa Indonesia
        processed_questions = []
        for q in questions:
            # Lowercase dan remove special characters
            q = re.sub(r'[^a-zA-Z0-9\s]', ' ', q.lower())
            q = re.sub(r'\s+', ' ', q).strip()
            processed_questions.append(q)
        
        # Fit TF-IDF vectorizer
        self.question_vectors = self.vectorizer.fit_transform(processed_questions)
        print(f"🔍 Similarity search ready with {self.question_vectors.shape[1]} features")
    
    def find_similar_questions(self, query: str, top_k: int = 3) -> List[Tuple[int, float, Dict]]:
        """Cari pertanyaan serupa menggunakan cosine similarity"""
        if self.question_vectors is None:
            return []
        
        # Preprocess query
        query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query.lower())
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Transform query ke vector
        query_vector = self.vectorizer.transform([query])
        
        # Hitung similarity
        similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.2:  # Threshold minimum similarity yang lebih tinggi
                results.append((
                    idx, 
                    similarities[idx], 
                    self.raw_data[idx]
                ))
        
        return results
    
    def train_model(self, epochs: int = 5, eval_split: float = 0.1):
        """Train model dengan parameter yang dioptimalkan"""
        if not self.dataset:
            print("❌ Dataset belum di-load!")
            return False
        
        print(f"🎯 Starting training with {len(self.dataset)} samples")
        
        # Split data untuk training dan evaluation
        dataset_size = len(self.dataset)
        eval_size = int(dataset_size * eval_split)
        train_size = dataset_size - eval_size
        
        train_dataset, eval_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, eval_size]
        )
        
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Eval samples: {len(eval_dataset)}")
        
        # Update training arguments
        self.training_args.num_train_epochs = epochs
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, bukan masked LM
            pad_to_multiple_of=8 if torch.cuda.is_available() else None
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        try:
            # Start training
            print("🚀 Training started...")
            trainer.train()
            
            # Save model
            model_path = "./optimized_motorcycle_model"
            trainer.save_model(model_path)
            self.tokenizer.save_pretrained(model_path)
            
            # Save training info
            training_info = {
                "model_name": self.model_name,
                "dataset_size": len(self.raw_data),
                "epochs": epochs,
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "vocab_size": len(self.tokenizer)
            }
            
            with open(f"{model_path}/training_info.json", "w", encoding='utf-8') as f:
                json.dump(training_info, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Training completed! Model saved to: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            return False
    
    def generate_answer(self, question: str, max_length: int = 150, use_similarity: bool = True) -> str:
        """Generate jawaban dengan flexible similarity search dan preprocessing bahasa gaul"""
        try:
            # Log user question
            user_msg_id = self.chat_manager.log_message(
                self.session_id, "user", question
            )
            
            similarity_score = 0.0
            context_used = "No context"
            
            # Preprocess question untuk handle bahasa gaul
            preprocessed_data = self.slang_preprocessor.preprocess_for_ai(question)
            normalized_question = preprocessed_data['normalized']
            slang_confidence = preprocessed_data['confidence']
            
            print(f"🔤 Original: {question}")
            print(f"🔄 Normalized: {normalized_question}")
            print(f"📊 Slang confidence: {slang_confidence:.2f}")
            
            # Gunakan flexible similarity search
            if use_similarity and hasattr(self, 'flexible_search'):
                try:
                    enhanced_response = self.flexible_search.get_enhanced_response(
                        question, custom_threshold=0.3
                    )
                    
                    if enhanced_response['similar_results']:
                        best_result = enhanced_response['similar_results'][0]
                        similarity_score = best_result['combined_similarity']
                        
                        # Jika similarity tinggi, gunakan reprocessed answer
                        if similarity_score > 0.6:
                            context_used = f"Flexible search - high similarity (score: {similarity_score:.3f})"
                            answer = enhanced_response['reprocessed_answer']
                            
                            # Log assistant response
                            self.chat_manager.log_message(
                                self.session_id, "assistant", answer, 
                                similarity_score, context_used
                            )
                            return answer
                        
                        # Jika similarity sedang, gunakan sebagai konteks untuk generation
                        elif similarity_score > 0.4:
                            context_parts = []
                            for result in enhanced_response['similar_results'][:2]:
                                if result['combined_similarity'] > 0.4:
                                    context_parts.append(f"Contoh: {result['question']} -> {result['answer']}")
                            
                            if context_parts:
                                context = "\n".join(context_parts)
                                context_used = f"Flexible search - medium similarity (score: {similarity_score:.3f})"
                                # Format prompt dengan konteks dari flexible search
                                input_text = f"""Anda adalah ahli sepeda motor. Berdasarkan contoh berikut:
{context}

Pertanyaan: {normalized_question}
Jawaban yang akurat dan relevan:"""
                            else:
                                context_used = "Flexible search - general context"
                                input_text = f"""Anda adalah ahli sepeda motor Indonesia.
Pertanyaan: {normalized_question}
Jawaban yang akurat:"""
                        else:
                            # Similarity rendah, gunakan reprocessed answer sebagai guidance
                            context_used = f"Flexible search - low similarity guidance (score: {similarity_score:.3f})"
                            guidance = enhanced_response['reprocessed_answer']
                            input_text = f"""Anda adalah ahli sepeda motor Indonesia. Berdasarkan informasi terkait:
{guidance}

Pertanyaan: {normalized_question}
Jawaban yang lebih spesifik:"""
                    
                except Exception as e:
                    print(f"⚠️ Flexible search error: {e}")
                    # Fallback ke similarity search lama
                    if self.raw_data:
                        similar_qa = self.find_similar_questions(question, top_k=3)
                        if similar_qa and similar_qa[0][1] > 0.4:
                            similarity_score = similar_qa[0][1]
                            context_used = f"Legacy similarity fallback (score: {similarity_score:.3f})"
                            input_text = f"""Anda adalah ahli sepeda motor Indonesia.
Pertanyaan: {normalized_question}
Jawaban yang akurat:"""
                        else:
                            context_used = "No similarity context"
                            input_text = f"""Anda adalah ahli sepeda motor Indonesia yang berpengalaman.
Pertanyaan: {normalized_question}
Jawaban yang informatif dan akurat:"""
                    else:
                        context_used = "Basic context"
                        input_text = f"""Anda adalah ahli sepeda motor Indonesia.
Pertanyaan: {normalized_question}
Jawaban:"""
            else:
                context_used = "Basic context - no flexible search"
                input_text = f"""Anda adalah ahli sepeda motor Indonesia.
Pertanyaan: {normalized_question}
Jawaban:"""
            
            # Tokenize input dengan batasan panjang
            inputs = self.tokenizer.encode(input_text, return_tensors='pt', max_length=400, truncation=True).to(self.device)
            
            # Generate response dengan parameter yang lebih konservatif
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.5,  # Lebih konservatif
                    do_sample=True,
                    top_p=0.8,  # Lebih fokus
                    top_k=40,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3  # Hindari pengulangan
                )
            
            # Decode dan clean response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract jawaban yang lebih baik
            if "Jawaban" in generated_text:
                # Ambil bagian setelah "Jawaban:"
                parts = generated_text.split("Jawaban")
                if len(parts) > 1:
                    answer = parts[-1].strip()
                    # Remove prefix seperti ":" atau "yang akurat:"
                    answer = re.sub(r'^[:\s]*(?:yang\s+[\w\s]+[:\s]*)?', '', answer)
                    # Clean up
                    answer = re.sub(r'<\|.*?\|>', '', answer)
                    answer = answer.replace('<|endoftext|>', '').strip()
                    
                    # Remove common artifacts
                    answer = re.sub(r'^[\s:]+', '', answer)  # Remove leading spaces and colons
                    answer = re.sub(r'\s+', ' ', answer)  # Normalize spaces
                    
                    # Validasi jawaban
                    if len(answer) > 15 and not answer.startswith("Pertanyaan:") and answer not in ["dan akurat:", "yang akurat:", ":"]:
                        # Potong jika terlalu panjang atau ada repetisi
                        sentences = answer.split('. ')
                        if len(sentences) > 3:
                            answer = '. '.join(sentences[:3]) + '.'
                        
                        # Log assistant response
                        self.chat_manager.log_message(
                            self.session_id, "assistant", answer, 
                            similarity_score, f"AI Generated - {context_used}"
                        )
                        return answer
            
            # Fallback ke similarity search jika generation gagal
            if use_similarity and self.raw_data:
                similar_qa = self.find_similar_questions(question, top_k=1)
                if similar_qa and similar_qa[0][1] > 0.3:
                    fallback_answer = similar_qa[0][2]['answer']
                    # Log fallback response
                    self.chat_manager.log_message(
                        self.session_id, "assistant", fallback_answer, 
                        similar_qa[0][1], "Fallback similarity match"
                    )
                    return fallback_answer
            
            # Log default response
            default_answer = "Maaf, saya belum memiliki informasi yang cukup untuk menjawab pertanyaan tentang motor tersebut. Bisa coba pertanyaan yang lebih spesifik?"
            self.chat_manager.log_message(
                self.session_id, "assistant", default_answer, 
                0.0, "Default response - no match found"
            )
            return default_answer
                
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            error_answer = "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda."
            # Log error response
            try:
                self.chat_manager.log_message(
                    self.session_id, "assistant", error_answer, 
                    0.0, f"Error response: {str(e)}"
                )
            except:
                pass  # Avoid nested errors
            return error_answer
    
    def interactive_mode(self):
        """Mode interaktif dengan adaptive learning"""
        print("\n🤖 Optimized Motorcycle AI - Interactive Mode with Adaptive Learning")
        print("Ketik 'quit' untuk keluar")
        print("Setelah setiap jawaban, Anda bisa memberikan feedback (y/n/skip)\n")
        
        conversation_count = 0
        last_message_id = None
        
        while True:
            try:
                question = input("❓ Pertanyaan: ").strip()
                
                if question.lower() in ['quit', 'exit', 'keluar']:
                    # Run auto learning cycle before exit
                    print("\n🧠 Menjalankan siklus pembelajaran adaptif...")
                    self.adaptive_system.auto_learning_cycle()
                    
                    # Show learning report
                    report = self.adaptive_system.get_learning_report()
                    print(f"\n📊 Laporan Pembelajaran:")
                    print(f"   - Total percakapan: {report['learning_stats']['total_conversations']}")
                    print(f"   - Feedback positif: {report['learning_stats']['positive_feedback']}")
                    print(f"   - Feedback negatif: {report['learning_stats']['negative_feedback']}")
                    print(f"   - Update model: {report['learning_stats']['model_updates']}")
                    
                    print("\n👋 Terima kasih! Sistem telah belajar dari percakapan ini.")
                    break
                
                if not question:
                    continue
                
                print("🤔 Memproses...")
                answer = self.generate_answer(question)
                print(f"🔧 Jawaban: {answer}")
                
                # Collect feedback
                try:
                    feedback = input("\n💭 Apakah jawaban ini membantu? (y=ya, n=tidak, s=skip): ").strip().lower()
                    
                    if feedback in ['y', 'yes', 'ya']:
                        # Get the last assistant message ID from chat history
                        recent_convs = self.chat_manager.get_recent_conversations(1)
                        if recent_convs and recent_convs[0]['message_type'] == 'assistant':
                            # Find the message ID (we'll need to modify chat_manager to return it)
                            self.chat_manager.add_feedback(
                                1, "positive", "User marked as helpful"
                            )
                        print("✅ Terima kasih atas feedback positif!")
                        
                    elif feedback in ['n', 'no', 'tidak']:
                        improvement = input("💡 Bagaimana jawaban bisa diperbaiki? (opsional): ").strip()
                        if recent_convs and recent_convs[0]['message_type'] == 'assistant':
                            self.chat_manager.add_feedback(
                                1, "negative", "User marked as not helpful", improvement
                            )
                        print("📝 Feedback negatif dicatat untuk pembelajaran.")
                        
                    elif feedback in ['s', 'skip']:
                        print("⏭️ Feedback dilewati.")
                        
                except KeyboardInterrupt:
                    print("\n⏭️ Feedback dilewati.")
                
                conversation_count += 1
                
                # Run auto learning every 5 conversations
                if conversation_count % 5 == 0:
                    print("\n🧠 Menjalankan pembelajaran adaptif...")
                    self.adaptive_system.auto_learning_cycle()
                    print("✅ Pembelajaran selesai.")
                
                print("\n" + "-"*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n🧠 Menjalankan siklus pembelajaran adaptif sebelum keluar...")
                self.adaptive_system.auto_learning_cycle()
                print("\n👋 Terima kasih!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

def main():
    """Main function untuk testing sistem"""
    print("🚀 Initializing Optimized Motorcycle AI...")
    
    # Initialize AI
    ai = OptimizedMotorcycleAI()
    
    # Load dataset
    dataset_path = "new_motorcycle_qa_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset tidak ditemukan: {dataset_path}")
        return
    
    print(f"📁 Loading dataset: {dataset_path}")
    if not ai.load_dataset(dataset_path):
        print("❌ Gagal load dataset!")
        return
    
    # Menu pilihan
    while True:
        print("\n" + "="*50)
        print("🤖 OPTIMIZED MOTORCYCLE AI SYSTEM")
        print("="*50)
        print("1. Train Model")
        print("2. Test AI (Interactive Mode)")
        print("3. Test Single Question")
        print("4. Exit")
        
        choice = input("\nPilih opsi (1-4): ").strip()
        
        if choice == '1':
            epochs = input("Jumlah epochs (default: 3): ").strip()
            epochs = int(epochs) if epochs.isdigit() else 3
            
            print(f"\n🎯 Starting training for {epochs} epochs...")
            ai.train_model(epochs=epochs)
            
        elif choice == '2':
            ai.interactive_mode()
            
        elif choice == '3':
            question = input("\nMasukkan pertanyaan: ").strip()
            if question:
                print("\n🤔 Memproses...")
                answer = ai.generate_answer(question)
                print(f"\n🔧 Jawaban: {answer}")
            
        elif choice == '4':
            print("👋 Terima kasih!")
            break
            
        else:
            print("❌ Pilihan tidak valid!")

if __name__ == "__main__":
    main()