import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
from datetime import datetime
import os

class FinalHybridMotorcycleAI:
    def __init__(self, model_path="improved_motorcycle_model_20250914_084736"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.conversation_log = []
        self.stats = {
            'rule_based': 0,
            'ai_generative': 0,
            'knowledge_base': 0,
            'total': 0
        }
        
        # Rule-based responses
        self.rules = {
            'greeting': {
                'patterns': [r'\b(hai|halo|hello|selamat)\b'],
                'response': "Halo! Saya adalah AI konsultan motor. Ada masalah dengan motor Anda yang bisa saya bantu?"
            },
            'thanks': {
                'patterns': [r'\b(terima kasih|thanks|makasih)\b'],
                'response': "Terima kasih kembali! Semoga motor Anda selalu dalam kondisi prima."
            },
            'goodbye': {
                'patterns': [r'\b(bye|sampai jumpa|selamat tinggal)\b'],
                'response': "Sampai jumpa! Jaga motor Anda dengan baik."
            }
        }
        
        # Enhanced knowledge base
        self.knowledge_base = {
            'susah_hidup': {
                'keywords': ['susah hidup', 'tidak nyala', 'mati', 'starter'],
                'response': "Motor susah hidup bisa disebabkan oleh: 1) Aki lemah/soak, 2) Busi kotor/aus, 3) Filter udara kotor, 4) Fuel pump bermasalah, 5) Sistem pengapian bermasalah. Coba cek aki dan busi terlebih dahulu.",
                'confidence': 0.9
            },
            'boros_bensin': {
                'keywords': ['boros', 'bensin habis', 'konsumsi tinggi'],
                'response': "Motor boros bensin dapat disebabkan: 1) Filter udara kotor, 2) Busi tidak optimal, 3) Injector kotor, 4) Tekanan ban kurang, 5) Gaya berkendara agresif. Lakukan service rutin dan cek komponen tersebut.",
                'confidence': 0.9
            },
            'asap_putih': {
                'keywords': ['asap putih', 'asap', 'knalpot'],
                'response': "Asap putih dari knalpot menandakan oli terbakar di ruang bakar. Penyebab: ring piston aus, seal klep rusak, atau head gasket bocor. Perlu pemeriksaan lebih lanjut dan kemungkinan overhaul mesin.",
                'confidence': 0.9
            },
            'getaran': {
                'keywords': ['bergetar', 'vibrasi', 'goyang'],
                'response': "Motor bergetar bisa karena: 1) Engine mounting rusak, 2) Balancing shaft bermasalah, 3) Busi mati, 4) Karburator tidak stel, 5) Rantai kendor. Cek engine mounting dan busi terlebih dahulu.",
                'confidence': 0.9
            },
            'rem_bermasalah': {
                'keywords': ['rem', 'tidak makan', 'blong', 'keras'],
                'response': "Masalah rem bisa disebabkan: 1) Kampas rem tipis, 2) Minyak rem kurang/kotor, 3) Kaliper macet, 4) Cakram aus/baret, 5) Udara dalam sistem rem. Segera perbaiki untuk keselamatan berkendara.",
                'confidence': 0.9
            },
            'overheat': {
                'keywords': ['panas', 'overheat', 'temperatur tinggi'],
                'response': "Motor overheat disebabkan: 1) Radiator kotor/bocor, 2) Coolant kurang, 3) Thermostat rusak, 4) Water pump bermasalah, 5) Kipas radiator mati. Segera matikan mesin dan biarkan dingin sebelum diperiksa.",
                'confidence': 0.9
            }
        }
        
        self.load_ai_model()
    
    def load_ai_model(self):
        """Load AI model"""
        try:
            print(f"Loading AI model from {self.model_path}...")
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.device = torch.device('cpu')  # Force CPU untuk stabilitas
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("✅ AI model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading AI model: {e}")
            self.model = None
    
    def check_rule_based(self, question):
        """Check rule-based responses"""
        question_lower = question.lower()
        
        for rule_type, rule_data in self.rules.items():
            for pattern in rule_data['patterns']:
                if re.search(pattern, question_lower):
                    return rule_data['response'], rule_type, 1.0
        
        return None, None, 0
    
    def check_knowledge_base(self, question):
        """Check knowledge base"""
        question_lower = question.lower()
        
        for category, data in self.knowledge_base.items():
            for keyword in data['keywords']:
                if keyword in question_lower:
                    return data['response'], category, data['confidence']
        
        return None, None, 0
    
    def generate_ai_response(self, question):
        """Generate AI response"""
        if not self.model:
            return None, 0
        
        try:
            # Format prompt
            prompt = f"Q: {question}\nA:"
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            inputs = inputs.to(self.device)
            
            # Generate with better parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.6,  # Lower temperature for more focused responses
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    top_p=0.9,
                    top_k=50
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response[len(prompt):].strip()
            
            # Clean and validate response
            cleaned_answer = self.clean_response(answer)
            
            if self.is_valid_response(cleaned_answer):
                return cleaned_answer, 0.8
            else:
                return None, 0
                
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return None, 0
    
    def clean_response(self, response):
        """Clean AI response"""
        if not response:
            return ""
        
        # Remove common artifacts
        response = re.sub(r'\b(A:|Q:)\b', '', response)
        response = re.sub(r'<\|.*?\|>', '', response)
        response = re.sub(r'\s+', ' ', response)
        
        # Take only first coherent sentence/paragraph
        sentences = response.split('.')
        if sentences:
            # Find first meaningful sentence
            for sentence in sentences[:3]:  # Check first 3 sentences
                sentence = sentence.strip()
                if len(sentence) > 20 and any(word in sentence.lower() for word in ['motor', 'mesin', 'oli', 'bensin', 'rem', 'ban']):
                    return sentence + '.'
        
        return response.strip()
    
    def is_valid_response(self, response):
        """Validate AI response quality"""
        if not response or len(response) < 15:
            return False
        
        # Check for nonsensical patterns
        nonsensical_patterns = [
            r'\b(untok|awp|perikshara|kondisi motor skep)\b',
            r'\b[A-Z]{3,}\b',  # Too many caps
            r'\d{6,}',  # Long numbers
            r'[^\w\s.,!?()-]{3,}'  # Strange characters
        ]
        
        for pattern in nonsensical_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return False
        
        # Check for motorcycle relevance
        motor_terms = ['motor', 'mesin', 'oli', 'bensin', 'rem', 'ban', 'aki', 'busi', 'karburator', 'injeksi']
        if not any(term in response.lower() for term in motor_terms):
            return False
        
        # Check word uniqueness
        words = response.lower().split()
        if len(set(words)) / len(words) < 0.6:  # Too repetitive
            return False
        
        return True
    
    def get_response(self, question):
        """Get best response using hybrid approach"""
        self.stats['total'] += 1
        
        # 1. Check rule-based (5%)
        rule_response, rule_type, rule_confidence = self.check_rule_based(question)
        if rule_response:
            self.stats['rule_based'] += 1
            return rule_response, 'rule_based', rule_type, rule_confidence
        
        # 2. Try AI generative (95% priority)
        if self.model:
            ai_response, ai_confidence = self.generate_ai_response(question)
            if ai_response and ai_confidence > 0.5:
                self.stats['ai_generative'] += 1
                return ai_response, 'ai_generative', 'ai_generated', ai_confidence
        
        # 3. Fallback to knowledge base
        kb_response, kb_category, kb_confidence = self.check_knowledge_base(question)
        if kb_response:
            self.stats['knowledge_base'] += 1
            return kb_response, 'knowledge_base', kb_category, kb_confidence
        
        # 4. Final fallback
        self.stats['knowledge_base'] += 1
        return "Maaf, saya perlu informasi lebih spesifik tentang masalah motor Anda. Bisa dijelaskan gejala yang dialami, jenis motor, dan kapan masalah terjadi?", 'fallback', 'general', 0.5
    
    def get_usage_statistics(self):
        """Get usage statistics"""
        if self.stats['total'] == 0:
            return "No queries processed yet."
        
        rule_pct = (self.stats['rule_based'] / self.stats['total']) * 100
        ai_pct = (self.stats['ai_generative'] / self.stats['total']) * 100
        kb_pct = (self.stats['knowledge_base'] / self.stats['total']) * 100
        
        return f"""📈 Usage Statistics:
Total Queries: {self.stats['total']}
Rule-based: {self.stats['rule_based']} ({rule_pct:.1f}%)
AI Generative: {self.stats['ai_generative']} ({ai_pct:.1f}%)
Knowledge Base: {self.stats['knowledge_base']} ({kb_pct:.1f}%)"""
    
    def save_conversation_log(self):
        """Save conversation log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"final_conversation_log_{timestamp}.json"
        
        log_data = {
            'timestamp': timestamp,
            'model_path': self.model_path,
            'statistics': self.stats,
            'conversations': self.conversation_log
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        return filename

def demo_mode():
    """Demo mode dengan pertanyaan preset"""
    ai = FinalHybridMotorcycleAI()
    
    demo_questions = [
        "Halo, selamat pagi!",
        "Motor saya susah hidup pagi-pagi",
        "Kenapa motor saya boros bensin ya?",
        "Motor keluar asap putih dari knalpot",
        "Motor saya bergetar saat idle",
        "Rem motor tidak makan",
        "Motor overheat terus",
        "Terima kasih atas bantuannya",
        "Bye, sampai jumpa"
    ]
    
    print("🚀 Final Hybrid Motorcycle AI - Demo Mode")
    print("=" * 60)
    
    for question in demo_questions:
        response, response_type, category, confidence = ai.get_response(question)
        
        print(f"\n❓ Q: {question}")
        print(f"🤖 A: {response}")
        print(f"📊 Type: {response_type} | Category: {category} | Confidence: {confidence}")
        print("-" * 80)
        
        # Log conversation
        ai.conversation_log.append({
            'question': question,
            'answer': response,
            'type': response_type,
            'category': category,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
    
    print(f"\n{ai.get_usage_statistics()}")
    
    # Save log
    log_file = ai.save_conversation_log()
    print(f"\n💾 Conversation log saved to: {log_file}")

def interactive_mode():
    """Interactive mode"""
    ai = FinalHybridMotorcycleAI()
    
    print("\n" + "=" * 60)
    print("🔧 Final Hybrid Motorcycle AI - Interactive Mode")
    print("=" * 60)
    print("Ketik 'quit' untuk keluar, 'stats' untuk statistik\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'keluar']:
                print(f"\n{ai.get_usage_statistics()}")
                log_file = ai.save_conversation_log()
                print(f"💾 Conversation log saved to: {log_file}")
                print("\n👋 Thank you for using Final Hybrid Motorcycle AI!")
                break
            
            if question.lower() == 'stats':
                print(f"\n{ai.get_usage_statistics()}\n")
                continue
            
            if not question:
                continue
            
            response, response_type, category, confidence = ai.get_response(question)
            print(f"🤖 A: {response}")
            print(f"📊 [{response_type}] Confidence: {confidence}\n")
            
            # Log conversation
            ai.conversation_log.append({
                'question': question,
                'answer': response,
                'type': response_type,
                'category': category,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
            
        except KeyboardInterrupt:
            print(f"\n\n{ai.get_usage_statistics()}")
            log_file = ai.save_conversation_log()
            print(f"💾 Conversation log saved to: {log_file}")
            print("\n👋 Thank you for using Final Hybrid Motorcycle AI!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    print("🚀 Final Hybrid Motorcycle AI System")
    print("Model: improved_motorcycle_model_20250914_081936")
    print("Architecture: 5% Rule-based + 95% AI Generative + Knowledge Base Fallback")
    
    mode = input("\nChoose mode - (d)emo or (i)nteractive: ").lower()
    
    if mode.startswith('d'):
        demo_mode()
    else:
        interactive_mode()

if __name__ == "__main__":
    main()