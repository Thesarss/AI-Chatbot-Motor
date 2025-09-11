import json
import re
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from context_manager import ContextManager, ConversationContext

class EnhancedMotorcycleAI:
    def __init__(self, model_path: str = None, dataset_path: str = None):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.context_manager = ContextManager()
        
        # Load model dan tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        # Topic keywords untuk deteksi topik
        self.topic_keywords = {
            "rem": ["rem", "brake", "kampas", "cakram", "tromol", "berhenti"],
            "mesin": ["mesin", "engine", "piston", "silinder", "kompresi", "oli"],
            "transmisi": ["transmisi", "gigi", "kopling", "cvt", "matic"],
            "kelistrikan": ["aki", "battery", "lampu", "starter", "alternator", "kabel"],
            "bahan_bakar": ["bensin", "fuel", "karburator", "injeksi", "tangki"],
            "suspensi": ["shock", "per", "suspensi", "ban", "velg"]
        }
    
    def detect_topic(self, text: str) -> Optional[str]:
        """Mendeteksi topik dari teks input"""
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        return None
    
    def extract_motorcycle_info(self, text: str) -> Dict[str, Any]:
        """Ekstrak informasi motor dari teks"""
        info = {}
        
        # Brand detection
        brands = ["honda", "yamaha", "suzuki", "kawasaki", "ducati", "harley"]
        for brand in brands:
            if brand in text.lower():
                info["brand"] = brand.title()
                break
        
        # Year detection
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            info["year"] = year_match.group()
        
        # Engine type detection
        if any(word in text.lower() for word in ["matic", "automatic", "cvt"]):
            info["engine_type"] = "Matic"
        elif any(word in text.lower() for word in ["manual", "kopling"]):
            info["engine_type"] = "Manual"
        
        return info
    
    def build_enhanced_prompt(self, user_input: str, context: ConversationContext) -> str:
        """Membangun prompt yang diperkaya dengan konteks"""
        base_prompt = f"""Anda adalah asisten AI ahli motor yang membantu mendiagnosis dan memberikan solusi masalah motor.

Konteks Percakapan:
{context.get_context_summary()}

Pertanyaan User: {user_input}

Berikan jawaban yang:
1. Sesuai dengan konteks percakapan sebelumnya
2. Spesifik untuk jenis motor yang sedang dibahas
3. Memberikan solusi step-by-step
4. Menyertakan estimasi biaya jika relevan
5. Tetap fokus pada topik yang sedang dibahas

Jawaban:"""
        
        return base_prompt
    
    def search_relevant_solutions(self, user_input: str, topic: str = None) -> List[Dict[str, Any]]:
        """Mencari solusi yang relevan dari dataset"""
        relevant_solutions = []
        input_lower = user_input.lower()
        
        for entry in self.dataset:
            # Cek kecocokan gejala
            gejala_match = any(keyword in input_lower for keyword in entry["gejala"].lower().split())
            
            # Cek kecocokan kategori dengan topik
            category_match = topic and entry.get("kategori", "").lower() == topic.lower()
            
            # Cek kecocokan kata kunci dalam permasalahan
            problem_match = any(keyword in entry["permasalahan"].lower() for keyword in input_lower.split())
            
            if gejala_match or category_match or problem_match:
                # Hitung skor relevansi
                score = 0
                if gejala_match: score += 3
                if category_match: score += 2
                if problem_match: score += 1
                
                entry_with_score = entry.copy()
                entry_with_score["relevance_score"] = score
                relevant_solutions.append(entry_with_score)
        
        # Sort berdasarkan skor relevansi
        relevant_solutions.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_solutions[:3]  # Ambil 3 solusi teratas
    
    def generate_response(self, user_input: str, session_id: str) -> str:
        """Generate respons dengan konteks yang diperkaya"""
        # Dapatkan atau buat konteks session
        context = self.context_manager.get_or_create_session(session_id)
        
        # Deteksi topik
        detected_topic = self.detect_topic(user_input)
        
        # Ekstrak informasi motor
        motor_info = self.extract_motorcycle_info(user_input)
        if motor_info:
            context.update_motorcycle_context(**motor_info)
        
        # Tambahkan pesan user ke konteks
        context.add_message(
            "user", 
            user_input, 
            {"topic": detected_topic} if detected_topic else {}
        )
        
        # Cari solusi yang relevan
        relevant_solutions = self.search_relevant_solutions(user_input, detected_topic)
        
        # Build enhanced prompt
        enhanced_prompt = self.build_enhanced_prompt(user_input, context)
        
        # Generate response menggunakan model
        inputs = self.tokenizer(enhanced_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ekstrak hanya bagian jawaban
        if "Jawaban:" in response:
            ai_response = response.split("Jawaban:")[-1].strip()
        else:
            ai_response = response.strip()
        
        # Jika model tidak memberikan jawaban yang baik, gunakan fallback
        if len(ai_response) < 50 or not ai_response:
            ai_response = self.generate_fallback_response(relevant_solutions, detected_topic)
        
        # Tambahkan respons AI ke konteks
        context.add_message("assistant", ai_response, {"topic": detected_topic})
        
        return ai_response
    
    def generate_fallback_response(self, relevant_solutions: List[Dict[str, Any]], topic: str = None) -> str:
        """Generate fallback response berdasarkan dataset"""
        if not relevant_solutions:
            return "Maaf, saya belum bisa memberikan solusi spesifik untuk masalah ini. Bisa Anda jelaskan lebih detail gejalanya?"
        
        best_solution = relevant_solutions[0]
        
        response = f"""Berdasarkan gejala yang Anda sebutkan, kemungkinan masalahnya adalah:

**Permasalahan:** {best_solution['permasalahan']}

**Solusi:**
{best_solution['solusi']}

**Estimasi Biaya:** {best_solution.get('estimasi_biaya', 'Tidak tersedia')}

**Kategori:** {best_solution.get('kategori', 'Umum')}"""
        
        if len(relevant_solutions) > 1:
            response += "\n\n**Kemungkinan lain:**\n"
            for i, sol in enumerate(relevant_solutions[1:], 1):
                response += f"{i}. {sol['permasalahan']} - {sol['solusi'][:100]}...\n"
        
        return response
    
    def get_session_context(self, session_id: str) -> Optional[ConversationContext]:
        """Mendapatkan konteks session"""
        return self.context_manager.active_sessions.get(session_id)
    
    def end_session(self, session_id: str, save_path: str = None):
        """Mengakhiri session"""
        self.context_manager.end_session(session_id, save_path)
    
    def cleanup_old_sessions(self):
        """Membersihkan session lama"""
        self.context_manager.cleanup_old_sessions()

# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi AI
    ai = EnhancedMotorcycleAI(
        model_path="./bengkelAI-Gemma-LORA-final",
        dataset_path="./otomotif_dataset_clean.json"
    )
    
    # Simulasi percakapan
    session_id = "user_123"
    
    # Percakapan 1
    response1 = ai.generate_response("Motor Honda Beat 2020 saya remnya blong", session_id)
    print("AI:", response1)
    
    # Percakapan 2 (masih dalam konteks rem)
    response2 = ai.generate_response("Berapa biaya untuk ganti kampas rem?", session_id)
    print("AI:", response2)
    
    # Percakapan 3 (ganti topik)
    response3 = ai.generate_response("Sekarang mesinnya susah hidup", session_id)
    print("AI:", response3)