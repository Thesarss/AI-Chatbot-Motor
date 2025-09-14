import json
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

class ImprovedHybridMotorcycleAI:
    def __init__(self, model_path="final_motorcycle_model_20250914_075414"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.ai_available = False
        self.load_ai_model()
        
        # Rule-based patterns (5% dari total respons)
        self.rule_patterns = {
            'greeting': {
                'patterns': [
                    r'\b(halo|hai|hello|selamat pagi|selamat siang|selamat sore|selamat malam)\b',
                    r'\b(hi|hey|good morning|good afternoon|good evening)\b'
                ],
                'responses': [
                    "Halo! Saya adalah asisten AI untuk konsultasi motor. Ada yang bisa saya bantu?",
                    "Hai! Selamat datang di layanan konsultasi motor. Silakan tanyakan masalah motor Anda.",
                    "Hello! Saya siap membantu Anda dengan masalah motor. Ada keluhan apa?"
                ]
            },
            'thanks': {
                'patterns': [
                    r'\b(terima kasih|thanks|thank you|makasih|thx)\b'
                ],
                'responses': [
                    "Sama-sama! Senang bisa membantu. Jangan ragu untuk bertanya lagi jika ada masalah motor lainnya.",
                    "Terima kasih kembali! Semoga motor Anda selalu dalam kondisi prima.",
                    "You're welcome! Jaga selalu kondisi motor Anda ya."
                ]
            },
            'goodbye': {
                'patterns': [
                    r'\b(bye|goodbye|selamat tinggal|sampai jumpa|dadah)\b'
                ],
                'responses': [
                    "Sampai jumpa! Jaga motor Anda dengan baik.",
                    "Selamat berkendara! Semoga perjalanan Anda aman.",
                    "Bye! Jangan lupa service rutin motor Anda."
                ]
            }
        }
        
        # Knowledge base untuk fallback (tetap AI-driven tapi dengan template)
        self.knowledge_base = {
            'susah_hidup': {
                'keywords': ['susah hidup', 'tidak mau hidup', 'susah nyala', 'tidak nyala'],
                'responses': [
                    "Motor susah hidup bisa disebabkan: 1) Busi kotor/rusak, 2) Filter udara kotor, 3) Karburator kotor, 4) Bahan bakar habis/kotor, 5) Aki lemah. Coba cek busi dan aki terlebih dahulu.",
                    "Penyebab motor susah hidup: Busi mati, karburator kotor, atau aki soak. Solusi: ganti busi, bersihkan karburator, charge aki. Estimasi biaya: Rp 50,000-200,000."
                ]
            },
            'boros_bensin': {
                'keywords': ['boros bensin', 'boros bbm', 'konsumsi tinggi'],
                'responses': [
                    "Motor boros bensin disebabkan: 1) Filter udara kotor, 2) Busi tidak sesuai spek, 3) Karburator tidak stel, 4) Tekanan ban kurang, 5) Gaya berkendara agresif. Coba service rutin dan cek tekanan ban.",
                    "Penyebab boros BBM: karburator kotor, filter udara tersumbat, atau riding style agresif. Solusi: tune up, ganti filter, berkendara halus. Bisa hemat 20-30%."
                ]
            },
            'asap_putih': {
                'keywords': ['asap putih', 'keluar asap putih', 'asap dari knalpot'],
                'responses': [
                    "Asap putih dari knalpot menandakan: 1) Oli masuk ruang bakar (ring piston aus), 2) Head gasket bocor, 3) Blok mesin retak. Ini masalah serius, segera ke bengkel untuk pengecekan lebih lanjut.",
                    "Asap putih = oli terbakar di ruang bakar. Penyebab: ring piston aus, seal klep rusak, atau head gasket bocor. Perlu overhaul mesin. Estimasi: Rp 1,500,000-3,000,000."
                ]
            },
            'getaran': {
                'keywords': ['getaran', 'bergetar', 'vibrasi'],
                'responses': [
                    "Motor bergetar bisa karena: 1) Engine mounting rusak, 2) Balancing shaft bermasalah, 3) Busi mati, 4) Karburator tidak stel, 5) Rantai kendor. Cek engine mounting dan busi dulu.",
                    "Getaran berlebih disebabkan: engine mounting aus, timing tidak tepat, atau komponen mesin tidak balance. Solusi: ganti mounting, stel timing, balancing mesin."
                ]
            }
        }
        
        self.usage_stats = {
            'rule_based': 0,
            'ai_generative': 0,
            'knowledge_base': 0,
            'total_queries': 0
        }
    
    def load_ai_model(self):
        """Load AI model untuk 95% respons generative"""
        try:
            print(f"Loading AI model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.ai_available = True
            print("✅ AI model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading AI model: {e}")
            print("Will use knowledge base and rule-based responses.")
            self.ai_available = False
    
    def check_rule_based(self, question):
        """Check if question matches rule-based patterns (5%)"""
        question_lower = question.lower()
        
        for category, data in self.rule_patterns.items():
            for pattern in data['patterns']:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    response = random.choice(data['responses'])
                    self.usage_stats['rule_based'] += 1
                    return True, response, category
        
        return False, None, None
    
    def check_knowledge_base(self, question):
        """Check knowledge base for common motorcycle issues"""
        question_lower = question.lower()
        
        for category, data in self.knowledge_base.items():
            for keyword in data['keywords']:
                if keyword in question_lower:
                    response = random.choice(data['responses'])
                    self.usage_stats['knowledge_base'] += 1
                    return True, response, category
        
        return False, None, None
    
    def generate_ai_response(self, question):
        """Generate response using AI model (95%)"""
        if not self.ai_available:
            return None
        
        try:
            # Simple prompt
            prompt = f"Q: {question}\nA:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True)
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Generate response dengan parameter konservatif
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.8,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer
            if "A:" in response:
                answer = response.split("A:")[-1].strip()
            else:
                answer = response.replace(prompt, "").strip()
            
            # Validate response quality
            if self.is_valid_response(answer):
                self.usage_stats['ai_generative'] += 1
                return answer
            else:
                return None
            
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return None
    
    def is_valid_response(self, response):
        """Validate if AI response is coherent"""
        if not response or len(response) < 20:
            return False
        
        # Check for repetitive patterns
        words = response.split()
        if len(words) < 8:
            return False
        
        # Check word diversity (more strict)
        unique_words = len(set(words))
        total_words = len(words)
        diversity_ratio = unique_words / total_words
        
        if diversity_ratio < 0.6:  # At least 60% unique words
            return False
        
        # Check for nonsensical repetitions
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return False
        
        # Check for common nonsensical patterns
        nonsensical_patterns = [
            'untok', 'mengatasinyaan', 'periksan', 'disregigant', 
            'tergiampol', 'ungguliki', 'agara', 'fairmotor',
            'adebabchen', 'ekling', 'berkai', 'remotor'
        ]
        
        response_lower = response.lower()
        for pattern in nonsensical_patterns:
            if pattern in response_lower:
                return False
        
        # Check if response contains meaningful motorcycle terms
        meaningful_terms = [
            'motor', 'mesin', 'oli', 'busi', 'karburator', 'rem', 
            'ban', 'rantai', 'gigi', 'service', 'perbaikan',
            'ganti', 'cek', 'periksa', 'rusak', 'aus'
        ]
        
        has_meaningful_term = any(term in response_lower for term in meaningful_terms)
        if not has_meaningful_term:
            return False
        
        return True
    
    def get_response(self, question):
        """Main method to get response (routing system)"""
        self.usage_stats['total_queries'] += 1
        
        # 1. Check rule-based first (greeting, thanks, etc.)
        is_rule_based, rule_response, category = self.check_rule_based(question)
        if is_rule_based:
            return {
                'response': rule_response,
                'type': 'rule_based',
                'category': category,
                'confidence': 1.0
            }
        
        # 2. Try AI generative (95% target)
        if self.ai_available:
            ai_response = self.generate_ai_response(question)
            if ai_response:
                return {
                    'response': ai_response,
                    'type': 'ai_generative',
                    'category': 'motorcycle_consultation',
                    'confidence': 0.85
                }
        
        # 3. Fallback to knowledge base
        is_kb, kb_response, kb_category = self.check_knowledge_base(question)
        if is_kb:
            return {
                'response': kb_response,
                'type': 'knowledge_base',
                'category': kb_category,
                'confidence': 0.9
            }
        
        # 4. Final fallback
        return {
            'response': "Maaf, saya perlu informasi lebih spesifik tentang masalah motor Anda. Bisa dijelaskan gejala yang dialami, jenis motor, dan kapan masalah terjadi?",
            'type': 'fallback',
            'category': 'general',
            'confidence': 0.5
        }
    
    def get_usage_statistics(self):
        """Get usage statistics"""
        total = self.usage_stats['total_queries']
        if total == 0:
            return self.usage_stats
        
        stats = self.usage_stats.copy()
        stats['rule_based_percentage'] = (self.usage_stats['rule_based'] / total) * 100
        stats['ai_generative_percentage'] = (self.usage_stats['ai_generative'] / total) * 100
        stats['knowledge_base_percentage'] = (self.usage_stats['knowledge_base'] / total) * 100
        
        return stats

def main():
    """Demo improved hybrid system"""
    print("🚀 Initializing Improved Hybrid Motorcycle AI System...")
    print("📊 Configuration: 5% Rule-based + 95% AI Generative (with KB fallback)")
    
    # Initialize system
    ai_system = ImprovedHybridMotorcycleAI()
    
    print("\n✅ System ready! Type 'quit' to exit.\n")
    
    # Test questions
    test_questions = [
        "Halo, saya butuh bantuan",
        "Motor saya susah hidup pagi hari",
        "Kenapa motor saya boros bensin?",
        "Motor keluar asap putih dari knalpot",
        "Motor saya bergetar saat idle",
        "Terima kasih atas bantuannya",
        "Bye, sampai jumpa"
    ]
    
    print("🧪 Testing with sample questions:\n")
    
    for question in test_questions:
        print(f"❓ Q: {question}")
        response_data = ai_system.get_response(question)
        print(f"🤖 A: {response_data['response']}")
        print(f"📊 Type: {response_data['type']} | Category: {response_data['category']} | Confidence: {response_data['confidence']}")
        print("-" * 80)
    
    # Show statistics
    stats = ai_system.get_usage_statistics()
    print("\n📈 Usage Statistics:")
    print(f"Total Queries: {stats['total_queries']}")
    print(f"Rule-based: {stats['rule_based']} ({stats.get('rule_based_percentage', 0):.1f}%)")
    print(f"AI Generative: {stats['ai_generative']} ({stats.get('ai_generative_percentage', 0):.1f}%)")
    print(f"Knowledge Base: {stats['knowledge_base']} ({stats.get('knowledge_base_percentage', 0):.1f}%)")
    
    # Interactive mode
    print("\n💬 Interactive mode (type 'quit' to exit):")
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question:
                response_data = ai_system.get_response(question)
                print(f"🤖 {response_data['response']}")
                print(f"📊 [{response_data['type']}] Confidence: {response_data['confidence']}")
        
        except KeyboardInterrupt:
            break
    
    # Final statistics
    final_stats = ai_system.get_usage_statistics()
    print("\n📊 Final Statistics:")
    print(f"Total Queries: {final_stats['total_queries']}")
    print(f"Rule-based: {final_stats['rule_based']} ({final_stats.get('rule_based_percentage', 0):.1f}%)")
    print(f"AI Generative: {final_stats['ai_generative']} ({final_stats.get('ai_generative_percentage', 0):.1f}%)")
    print(f"Knowledge Base: {final_stats['knowledge_base']} ({final_stats.get('knowledge_base_percentage', 0):.1f}%)")
    print("\n👋 Thank you for using Improved Hybrid Motorcycle AI!")

if __name__ == "__main__":
    main()