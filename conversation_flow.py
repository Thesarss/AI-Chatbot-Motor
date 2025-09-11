from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
from context_scorer import ContextScorer

class ConversationState(Enum):
    GREETING = "greeting"
    PROBLEM_DESCRIPTION = "problem_description"
    PROBLEM_IDENTIFICATION = "problem_identification"
    DIAGNOSIS = "diagnosis"
    SOLUTION = "solution"
    FOLLOW_UP = "follow_up"
    CLOSING = "closing"

class ConversationFlow:
    def __init__(self):
        self.context_scorer = ContextScorer()
        
        # State transition rules
        self.state_transitions = {
            ConversationState.GREETING: [
                ConversationState.PROBLEM_IDENTIFICATION,
                ConversationState.DIAGNOSIS
            ],
            ConversationState.PROBLEM_IDENTIFICATION: [
                ConversationState.DIAGNOSIS,
                ConversationState.SOLUTION
            ],
            ConversationState.DIAGNOSIS: [
                ConversationState.SOLUTION,
                ConversationState.PROBLEM_IDENTIFICATION
            ],
            ConversationState.SOLUTION: [
                ConversationState.FOLLOW_UP,
                ConversationState.CLOSING
            ],
            ConversationState.FOLLOW_UP: [
                ConversationState.DIAGNOSIS,
                ConversationState.SOLUTION,
                ConversationState.CLOSING
            ],
            ConversationState.CLOSING: [
                ConversationState.GREETING,
                ConversationState.PROBLEM_IDENTIFICATION
            ]
        }
        
        # Response templates untuk setiap state
        self.response_templates = {
            ConversationState.GREETING: {
                "prompts": [
                    "Halo! Saya asisten AI untuk masalah motor. Ada masalah apa dengan motor Anda?",
                    "Selamat datang! Ceritakan masalah motor yang Anda alami."
                ],
                "follow_up_questions": [
                    "Bisa ceritakan gejala yang Anda rasakan?",
                    "Motor apa yang Anda gunakan?"
                ]
            },
            ConversationState.PROBLEM_IDENTIFICATION: {
                "prompts": [
                    "Untuk membantu diagnosis yang tepat, bisa Anda jelaskan:",
                    "Agar saya bisa membantu lebih baik, tolong berikan informasi:"
                ],
                "follow_up_questions": [
                    "Kapan masalah ini mulai terjadi?",
                    "Apakah ada bunyi atau gejala khusus?",
                    "Sudah berapa lama motor digunakan?",
                    "Apakah masalah terjadi saat kondisi tertentu?"
                ]
            },
            ConversationState.DIAGNOSIS: {
                "prompts": [
                    "Berdasarkan gejala yang Anda sebutkan, kemungkinan masalahnya adalah:",
                    "Dari informasi yang Anda berikan, saya menganalisis bahwa:"
                ],
                "follow_up_questions": [
                    "Apakah diagnosis ini sesuai dengan yang Anda rasakan?",
                    "Adakah gejala lain yang belum disebutkan?"
                ]
            },
            ConversationState.SOLUTION: {
                "prompts": [
                    "Berikut langkah-langkah untuk mengatasi masalah ini:",
                    "Solusi yang bisa Anda lakukan:"
                ],
                "follow_up_questions": [
                    "Apakah Anda ingin tahu estimasi biayanya?",
                    "Adakah langkah yang kurang jelas?",
                    "Apakah Anda bisa melakukan perbaikan sendiri?"
                ]
            },
            ConversationState.FOLLOW_UP: {
                "prompts": [
                    "Apakah ada hal lain yang ingin Anda tanyakan?",
                    "Adakah masalah motor lain yang perlu dibahas?"
                ],
                "follow_up_questions": [
                    "Bagaimana dengan komponen motor lainnya?",
                    "Apakah ada perawatan rutin yang ingin ditanyakan?"
                ]
            }
        }
        
        # Intent to state mapping
        self.intent_state_mapping = {
            "diagnosis": ConversationState.DIAGNOSIS,
            "solution": ConversationState.SOLUTION,
            "prevention": ConversationState.FOLLOW_UP
        }
    
    def determine_conversation_state(self, 
                                   user_input: str, 
                                   conversation_history: List[Dict[str, Any]], 
                                   current_state: ConversationState = None) -> ConversationState:
        """Menentukan state percakapan berdasarkan input dan history"""
        
        # Jika belum ada state, mulai dari greeting
        if not current_state and not conversation_history:
            return ConversationState.GREETING
        
        # Analisis intent dari user input
        scores = self.context_scorer.calculate_context_relevance(
            user_input, conversation_history
        )
        detected_intent = scores.get("detected_intent", "unknown")
        
        # Mapping intent ke state
        if detected_intent in self.intent_state_mapping:
            target_state = self.intent_state_mapping[detected_intent]
            
            # Cek apakah transisi valid
            if current_state and target_state in self.state_transitions.get(current_state, []):
                return target_state
            elif not current_state:
                return target_state
        
        # Deteksi perubahan topik
        if self.is_topic_change(user_input, conversation_history):
            return ConversationState.PROBLEM_IDENTIFICATION
        
        # Deteksi greeting patterns
        greeting_patterns = ["halo", "hai", "selamat", "permisi", "assalamualaikum"]
        if any(pattern in user_input.lower() for pattern in greeting_patterns):
            return ConversationState.GREETING
        
        # Deteksi closing patterns
        closing_patterns = ["terima kasih", "selesai", "cukup", "sampai jumpa", "bye"]
        if any(pattern in user_input.lower() for pattern in closing_patterns):
            return ConversationState.CLOSING
        
        # Default state progression
        if current_state:
            return self.get_next_logical_state(current_state, user_input)
        
        return ConversationState.PROBLEM_IDENTIFICATION
    
    def is_topic_change(self, user_input: str, conversation_history: List[Dict[str, Any]]) -> bool:
        """Mendeteksi apakah ada perubahan topik"""
        if not conversation_history:
            return False
        
        # Ambil topik dari pesan terakhir
        last_topic = None
        for message in reversed(conversation_history[-3:]):
            topic = message.get("metadata", {}).get("topic")
            if topic:
                last_topic = topic
                break
        
        if not last_topic:
            return False
        
        # Deteksi topik dari input saat ini
        current_scores = {}
        for category in self.context_scorer.category_keywords.keys():
            score = self.context_scorer.calculate_keyword_score(user_input, category)
            current_scores[category] = score
        
        if current_scores:
            current_topic = max(current_scores, key=current_scores.get)
            # Jika skor topik baru cukup tinggi dan berbeda dari topik sebelumnya
            if current_scores[current_topic] > 0.3 and current_topic != last_topic:
                return True
        
        return False
    
    def get_next_logical_state(self, current_state: ConversationState, user_input: str) -> ConversationState:
        """Mendapatkan state logis berikutnya"""
        # Analisis user input untuk menentukan arah percakapan
        solution_keywords = ["bagaimana", "cara", "solusi", "perbaiki", "atasi"]
        diagnosis_keywords = ["kenapa", "mengapa", "penyebab", "masalah"]
        
        user_lower = user_input.lower()
        
        if any(kw in user_lower for kw in solution_keywords):
            return ConversationState.SOLUTION
        elif any(kw in user_lower for kw in diagnosis_keywords):
            return ConversationState.DIAGNOSIS
        
        # Default progression berdasarkan state saat ini
        transitions = self.state_transitions.get(current_state, [])
        if transitions:
            return transitions[0]  # Ambil transisi pertama sebagai default
        
        return current_state
    
    def generate_contextual_prompt(self, 
                                 state: ConversationState, 
                                 user_input: str, 
                                 conversation_history: List[Dict[str, Any]]) -> str:
        """Generate prompt yang sesuai dengan state percakapan"""
        
        template = self.response_templates.get(state, {})
        prompts = template.get("prompts", ["Saya akan membantu Anda."])
        follow_ups = template.get("follow_up_questions", [])
        
        # Pilih prompt yang sesuai
        base_prompt = prompts[0] if prompts else "Saya akan membantu Anda."
        
        # Tambahkan konteks dari conversation history
        context_info = self.extract_context_info(conversation_history)
        
        # Build enhanced prompt
        enhanced_prompt = f"""
Anda adalah asisten AI ahli motor. State percakapan saat ini: {state.value}

Konteks percakapan:
{context_info}

User input: {user_input}

Instruksi respons:
1. {base_prompt}
2. Berikan jawaban yang spesifik dan actionable
3. Pertahankan konsistensi dengan percakapan sebelumnya
4. Gunakan bahasa yang mudah dipahami
"""
        
        # Tambahkan follow-up questions jika sesuai
        if follow_ups and state in [ConversationState.PROBLEM_IDENTIFICATION, ConversationState.FOLLOW_UP]:
            enhanced_prompt += f"\n5. Ajukan pertanyaan follow-up yang relevan dari: {', '.join(follow_ups[:2])}"
        
        enhanced_prompt += "\n\nJawaban:"
        
        return enhanced_prompt
    
    def extract_context_info(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Ekstrak informasi konteks dari history percakapan"""
        if not conversation_history:
            return "Percakapan baru dimulai."
        
        context_parts = []
        
        # Ekstrak topik yang dibahas
        topics = set()
        for message in conversation_history[-5:]:
            topic = message.get("metadata", {}).get("topic")
            if topic:
                topics.add(topic)
        
        if topics:
            context_parts.append(f"Topik yang dibahas: {', '.join(topics)}")
        
        # Ekstrak informasi motor jika ada
        motor_info = []
        for message in conversation_history:
            content = message.get("content", "").lower()
            # Simple extraction - bisa diperbaiki dengan NER
            brands = ["honda", "yamaha", "suzuki", "kawasaki"]
            for brand in brands:
                if brand in content:
                    motor_info.append(f"Brand: {brand.title()}")
                    break
        
        if motor_info:
            context_parts.append(motor_info[0])
        
        # Ringkasan percakapan terakhir
        if len(conversation_history) >= 2:
            last_user = ""
            last_ai = ""
            
            for message in reversed(conversation_history[-4:]):
                if message["role"] == "user" and not last_user:
                    last_user = message["content"][:100]
                elif message["role"] == "assistant" and not last_ai:
                    last_ai = message["content"][:100]
                
                if last_user and last_ai:
                    break
            
            if last_user:
                context_parts.append(f"Pertanyaan terakhir: {last_user}...")
            if last_ai:
                context_parts.append(f"Respons terakhir: {last_ai}...")
        
        return " | ".join(context_parts) if context_parts else "Konteks percakapan sedang dibangun."
    
    def should_ask_clarification(self, 
                               user_input: str, 
                               conversation_history: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Menentukan apakah perlu klarifikasi dan pertanyaan apa"""
        
        # Cek jika input terlalu pendek atau tidak jelas
        if len(user_input.strip()) < 10:
            return True, "Bisa Anda jelaskan lebih detail masalah yang dialami?"
        
        # Cek jika tidak ada keyword motor yang terdeteksi
        scores = self.context_scorer.calculate_context_relevance(
            user_input, conversation_history
        )
        
        if scores.get("keyword_match", 0) < 0.2:
            return True, "Bisa Anda sebutkan komponen motor mana yang bermasalah? (contoh: rem, mesin, transmisi)"
        
        # Cek jika intent tidak jelas
        if scores.get("user_intent", 0) < 0.3:
            return True, "Apakah Anda ingin mengetahui penyebab masalah, cara perbaikan, atau estimasi biaya?"
        
        return False, ""
    
    def get_conversation_summary(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Membuat ringkasan percakapan"""
        if not conversation_history:
            return {"status": "empty", "summary": "Belum ada percakapan"}
        
        # Hitung statistik
        user_messages = [msg for msg in conversation_history if msg["role"] == "user"]
        ai_messages = [msg for msg in conversation_history if msg["role"] == "assistant"]
        
        # Ekstrak topik yang dibahas
        topics_discussed = set()
        for message in conversation_history:
            topic = message.get("metadata", {}).get("topic")
            if topic:
                topics_discussed.add(topic)
        
        # Tentukan status percakapan
        last_message = conversation_history[-1] if conversation_history else None
        if last_message and "terima kasih" in last_message.get("content", "").lower():
            status = "completed"
        elif len(user_messages) > 3:
            status = "ongoing"
        else:
            status = "starting"
        
        return {
            "status": status,
            "total_messages": len(conversation_history),
            "user_messages": len(user_messages),
            "ai_messages": len(ai_messages),
            "topics_discussed": list(topics_discussed),
            "summary": f"Percakapan {status} dengan {len(topics_discussed)} topik dibahas"
        }

# Contoh penggunaan
if __name__ == "__main__":
    flow = ConversationFlow()
    
    # Simulasi percakapan
    conversation_history = []
    current_state = None
    
    # Input 1
    user_input1 = "Halo, motor saya remnya blong"
    current_state = flow.determine_conversation_state(user_input1, conversation_history, current_state)
    prompt1 = flow.generate_contextual_prompt(current_state, user_input1, conversation_history)
    
    print(f"State: {current_state}")
    print(f"Prompt: {prompt1}")
    print("-" * 50)
    
    # Tambahkan ke history
    conversation_history.append({
        "role": "user",
        "content": user_input1,
        "metadata": {"topic": "rem"}
    })
    
    # Input 2
    user_input2 = "Berapa biaya untuk perbaikannya?"
    current_state = flow.determine_conversation_state(user_input2, conversation_history, current_state)
    prompt2 = flow.generate_contextual_prompt(current_state, user_input2, conversation_history)
    
    print(f"State: {current_state}")
    print(f"Prompt: {prompt2}")