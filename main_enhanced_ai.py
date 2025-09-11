# File: main_enhanced_ai.py (Revisi Final)

import json
import re  # ✅ Import modul 're' yang sebelumnya hilang
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss

# ✅ Mengimpor kelas yang benar dari context_manager.py
from context_manager import ContextManager, ConversationContext

# ✅ Mengganti nama kelas agar konsisten dengan skrip lain
class EnhancedMotorcycleAssistant:
    def __init__(self, model_path: str, dataset_path: str = "otomotif_dataset_clean.json"):
        """
        Inisialisasi AI dengan memuat base model, adapter LoRA, dan retriever RAG.
        """
        self.lora_adapter_path = model_path
        self.dataset_path = dataset_path
        self.context_manager = ContextManager()

        self._setup_model()
        self._setup_rag()

    def _setup_model(self):
        """Memuat base model dan menggabungkannya dengan adapter LoRA."""
        base_model_name = "google/gemma-2b-it"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔥 Memuat model di device: {device}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        print(f"🔗 Menggabungkan dengan adapter LoRA dari: {self.lora_adapter_path}...")
        self.model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
        print("✅ Model berhasil dimuat.")

    def _setup_rag(self):
        """Mempersiapkan retriever untuk RAG."""
        print("🔧 Mempersiapkan retriever RAG...")
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
            
            knowledge_base = [item['instruction'] for item in self.dataset]
            self.retriever_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            
            print("Embedding dataset untuk pencarian cepat...")
            knowledge_embeddings = self.retriever_model.encode(knowledge_base, convert_to_tensor=True)
            self.index = faiss.IndexFlatL2(knowledge_embeddings.shape[1])
            self.index.add(knowledge_embeddings.cpu().numpy())
            print("✅ Retriever RAG siap.")
        except FileNotFoundError:
            print(f"❌ File dataset '{self.dataset_path}' tidak ditemukan. RAG akan dinonaktifkan.")
            self.dataset = []
            self.index = None

    def detect_topic(self, text: str) -> Optional[str]:
        """Mendeteksi topik dari teks input menggunakan RAG retriever (lebih akurat)."""
        if not self.index:
            return "umum"
        
        query_embedding = self.retriever_model.encode(text, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        _, indices = self.index.search(query_embedding, 1)
        
        if len(indices[0]) > 0:
            best_match = self.dataset[indices[0][0]]
            return best_match.get("category", "umum")
        return "umum"

    def extract_motorcycle_info(self, text: str) -> Dict[str, Any]:
        """Ekstrak informasi motor dari teks."""
        info = {}
        text_lower = text.lower()
        brands = ["honda", "yamaha", "suzuki", "kawasaki", "vario", "nmax", "beat", "ninja"]
        for brand in brands:
            if brand in text_lower:
                info["brand/model"] = brand.title()
                break
        
        year_match = re.search(r'\b(201[5-9]|202[0-5])\b', text)
        if year_match:
            info["year"] = year_match.group()
        return info

    def build_enhanced_prompt(self, user_input: str, context: ConversationContext, rag_context_str: str) -> str:
        """Membangun prompt yang diperkaya dengan konteks untuk Gemma."""
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context.conversation_history[-5:]])
        
        instruction = (
            "Anda adalah asisten AI bengkel yang ahli dan membantu. "
            "Gunakan konteks RAG dan riwayat percakapan untuk memberikan jawaban yang relevan dan akurat. "
            "Jika ini adalah diagnosis masalah baru, jawab dalam format 'Permasalahan:', 'Solusi:', dan 'Estimasi Biaya:'. "
            "Jika ini pertanyaan lanjutan, jawab secara natural berdasarkan histori."
        )

        chat = [
            {"role": "user", "content": f"## Instruksi Sistem\n{instruction}\n\n## Konteks dari Database (RAG)\n{rag_context_str}\n\n## Riwayat Percakapan\n{history_str}\n\n## Pertanyaan Baru dari Pengguna\n{user_input}"}
        ]
        
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def search_relevant_solutions(self, user_input: str) -> str:
        """Mencari solusi relevan menggunakan RAG dan mengformatnya menjadi string."""
        if not self.index:
            return "Database tidak tersedia."

        query_embedding = self.retriever_model.encode(user_input, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        _, indices = self.index.search(query_embedding, 3)
        
        context_str = "Berikut adalah beberapa data yang relevan dari database:\n"
        for i in indices[0]:
            item = self.dataset[i]
            context_str += (
                f"- Keluhan: {item['instruction']}\n"
                f"  Jawaban: Permasalahan: {item['response']['permasalahan']} | Solusi: {item['response']['solusi']} | Estimasi Biaya: {item['response']['biaya']}\n\n"
            )
        return context_str

    def process_message(self, user_input: str, session_id: str) -> str:
        """Fungsi utama untuk memproses input dan menghasilkan respons."""
        # ✅ Menggunakan get_or_create_session dari ContextManager yang benar
        context = self.context_manager.get_or_create_session(session_id)
        
        motor_info = self.extract_motorcycle_info(user_input)
        if motor_info:
            context.update_motorcycle_context(**motor_info)
        
        is_new_topic = len(context.conversation_history) == 0
        
        rag_context_str = ""
        if is_new_topic:
            rag_context_str = self.search_relevant_solutions(user_input)
            detected_topic = self.detect_topic(user_input)
            context.current_topic = detected_topic
            context.rag_context = rag_context_str # Simpan RAG konteks di sesi
        else:
            rag_context_str = getattr(context, 'rag_context', '') # Gunakan RAG konteks dari awal sesi
        
        self.context_manager.add_message(session_id, "user", user_input)

        enhanced_prompt = self.build_enhanced_prompt(user_input, context, rag_context_str)
        
        inputs = self.tokenizer(enhanced_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        ai_response = response.split("<start_of_turn>model\n")[-1].strip()
        
        self.context_manager.add_message(session_id, "assistant", ai_response)
        
        return ai_response

# ===============================
# Contoh Penggunaan jika file ini dijalankan langsung
# ===============================
if __name__ == "__main__":
    print("Menginisialisasi AI untuk demo mandiri...")
    # Anda perlu sudah login ke Hugging Face sebelum menjalankan ini
    # Gunakan `huggingface-cli login` di terminal
    
    try:
        # ✅ Nama kelas diubah menjadi EnhancedMotorcycleAssistant
        ai = EnhancedMotorcycleAssistant(
            model_path="./bengkelAI-Gemma-LORA-final",
            dataset_path="otomotif_dataset_clean.json"
        )
        
        session_id = "user_demo_123"
        
        print("\n--- Sesi Percakapan Demo ---")
        
        # Percakapan 1 (Masalah baru)
        user_q1 = "Halo, motor Honda Vario 2020 saya gasnya terasa berat dan brebet"
        print(f"\n👤 Anda: {user_q1}")
        response1 = ai.process_message(user_q1, session_id)
        print(f"🤖 BengkelAI:\n{response1}")

        # Pertanyaan lanjutan
        user_q2 = "Apakah saya bisa membersihkan filternya sendiri?"
        print(f"\n👤 Anda: {user_q2}")
        response2 = ai.process_message(user_q2, session_id)
        print(f"🤖 BengkelAI:\n{response2}")
        
    except Exception as e:
        print(f"\n❌ Terjadi error saat inisialisasi atau demo: {e}")
        print("💡 Pastikan Anda sudah login ke Hugging Face dan path model sudah benar.")