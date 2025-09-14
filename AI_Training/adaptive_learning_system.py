import json
import os
from datetime import datetime
from typing import List, Dict, Any
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
from difflib import SequenceMatcher

class AdaptiveLearningSystem:
    def __init__(self, model_path: str, chat_history_file: str = "chat_history.json"):
        self.model_path = model_path
        self.chat_history_file = chat_history_file
        self.learning_threshold = 0.7  # Similarity threshold for learning
        self.min_interactions = 3  # Minimum interactions before learning
        
        # Initialize chat history storage
        self.chat_history = self.load_chat_history()
        
        print("🤖 Adaptive Learning System initialized!")
    
    def load_chat_history(self) -> List[Dict]:
        """Load existing chat history"""
        if os.path.exists(self.chat_history_file):
            try:
                with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_chat_history(self):
        """Save chat history to file"""
        with open(self.chat_history_file, 'w', encoding='utf-8') as f:
            json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
    
    def add_interaction(self, user_input: str, ai_response: str, user_feedback: str = None):
        """Add new chat interaction"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input.strip(),
            "ai_response": ai_response.strip(),
            "user_feedback": user_feedback,
            "interaction_id": hashlib.md5(f"{user_input}{ai_response}".encode()).hexdigest()[:8]
        }
        
        self.chat_history.append(interaction)
        self.save_chat_history()
        print(f"💬 Interaction saved: {interaction['interaction_id']}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def find_similar_interactions(self, user_input: str) -> List[Dict]:
        """Find similar past interactions"""
        similar = []
        for interaction in self.chat_history:
            similarity = self.calculate_similarity(user_input, interaction['user_input'])
            if similarity > self.learning_threshold:
                interaction['similarity'] = similarity
                similar.append(interaction)
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)
    
    def extract_learning_patterns(self) -> List[Dict]:
        """Extract learning patterns from chat history"""
        patterns = []
        
        # Group similar questions
        question_groups = {}
        for interaction in self.chat_history:
            user_input = interaction['user_input']
            found_group = False
            
            for group_key in question_groups:
                if self.calculate_similarity(user_input, group_key) > 0.6:
                    question_groups[group_key].append(interaction)
                    found_group = True
                    break
            
            if not found_group:
                question_groups[user_input] = [interaction]
        
        # Extract patterns from groups with multiple interactions
        for group_key, interactions in question_groups.items():
            if len(interactions) >= self.min_interactions:
                # Find the best response based on user feedback
                best_response = self.find_best_response(interactions)
                if best_response:
                    patterns.append({
                        "question_pattern": group_key,
                        "best_response": best_response,
                        "interaction_count": len(interactions),
                        "confidence": min(1.0, len(interactions) / 10.0)
                    })
        
        return patterns
    
    def find_best_response(self, interactions: List[Dict]) -> str:
        """Find the best response from multiple interactions"""
        # Prioritize responses with positive feedback
        positive_responses = [i for i in interactions if i.get('user_feedback') == 'positive']
        if positive_responses:
            return positive_responses[-1]['ai_response']  # Latest positive response
        
        # If no feedback, return the latest response
        return interactions[-1]['ai_response']
    
    def generate_training_data_from_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Generate training data from learned patterns"""
        training_data = []
        
        for pattern in patterns:
            # Create training example
            training_example = {
                "instruction": "Jawab pertanyaan tentang motor berikut dengan akurat:",
                "input": pattern['question_pattern'],
                "output": pattern['best_response']
            }
            training_data.append(training_example)
            
            # Generate variations of the question
            variations = self.generate_question_variations(pattern['question_pattern'])
            for variation in variations:
                training_data.append({
                    "instruction": "Jawab pertanyaan tentang motor berikut dengan akurat:",
                    "input": variation,
                    "output": pattern['best_response']
                })
        
        return training_data
    
    def generate_question_variations(self, question: str) -> List[str]:
        """Generate variations of a question"""
        variations = []
        
        # Simple variations by changing question words
        question_words = {
            "bagaimana": ["gimana", "cara"],
            "kenapa": ["mengapa", "kok"],
            "apa": ["apakah"],
            "dimana": ["di mana"],
            "kapan": ["bilamana"]
        }
        
        for original, replacements in question_words.items():
            if original in question.lower():
                for replacement in replacements:
                    variation = question.lower().replace(original, replacement)
                    variations.append(variation.capitalize())
        
        return variations[:3]  # Limit to 3 variations
    
    def create_adaptive_dataset(self, base_dataset_path: str, output_path: str):
        """Create adaptive dataset combining base data and learned patterns"""
        print("🧠 Creating adaptive dataset from chat history...")
        
        # Load base dataset
        with open(base_dataset_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # Extract learning patterns
        patterns = self.extract_learning_patterns()
        print(f"📊 Found {len(patterns)} learning patterns")
        
        # Generate training data from patterns
        adaptive_data = self.generate_training_data_from_patterns(patterns)
        print(f"🎯 Generated {len(adaptive_data)} adaptive training examples")
        
        # Combine datasets
        combined_data = base_data + adaptive_data
        
        # Save adaptive dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Adaptive dataset saved: {output_path}")
        print(f"📈 Total entries: {len(combined_data)} (Base: {len(base_data)}, Adaptive: {len(adaptive_data)})")
        
        return output_path
    
    def get_contextual_response(self, user_input: str) -> Dict:
        """Get contextual response based on chat history"""
        similar_interactions = self.find_similar_interactions(user_input)
        
        if similar_interactions:
            best_match = similar_interactions[0]
            return {
                "has_context": True,
                "suggested_response": best_match['ai_response'],
                "confidence": best_match['similarity'],
                "interaction_id": best_match['interaction_id']
            }
        
        return {
            "has_context": False,
            "suggested_response": None,
            "confidence": 0.0,
            "interaction_id": None
        }
    
    def generate_learning_report(self) -> Dict:
        """Generate learning report"""
        patterns = self.extract_learning_patterns()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_interactions": len(self.chat_history),
            "learning_patterns": len(patterns),
            "avg_confidence": sum(p['confidence'] for p in patterns) / len(patterns) if patterns else 0,
            "top_patterns": sorted(patterns, key=lambda x: x['confidence'], reverse=True)[:5]
        }
        
        return report

def main():
    # Initialize adaptive learning system
    model_path = "improved_motorcycle_model_20250914_074059"
    learning_system = AdaptiveLearningSystem(model_path)
    
    # Example: Add some sample interactions
    sample_interactions = [
        {
            "user_input": "Motor saya susah hidup, kenapa ya?",
            "ai_response": "Motor susah hidup bisa disebabkan beberapa hal: 1) Aki lemah atau habis, 2) Busi kotor atau rusak, 3) Bensin habis atau kotor, 4) Filter udara tersumbat. Coba cek aki dan busi terlebih dahulu.",
            "user_feedback": "positive"
        },
        {
            "user_input": "Gimana cara ganti oli motor?",
            "ai_response": "Cara ganti oli motor: 1) Panaskan mesin sebentar, 2) Matikan mesin dan tunggu dingin, 3) Buka baut pembuangan oli, 4) Tunggu oli lama keluar semua, 5) Pasang kembali baut pembuangan, 6) Isi oli baru sesuai kapasitas, 7) Cek level oli dengan dipstick.",
            "user_feedback": "positive"
        },
        {
            "user_input": "Ban motor saya gundul, bahaya gak?",
            "ai_response": "Ban gundul sangat berbahaya! Risiko: 1) Mudah selip saat hujan, 2) Jarak pengereman lebih jauh, 3) Handling motor tidak stabil, 4) Bisa pecah saat berkendara. Segera ganti ban baru untuk keselamatan.",
            "user_feedback": "positive"
        }
    ]
    
    # Add sample interactions
    for interaction in sample_interactions:
        learning_system.add_interaction(
            interaction["user_input"],
            interaction["ai_response"],
            interaction["user_feedback"]
        )
    
    # Create adaptive dataset
    base_dataset = "ultra_clean_motorcycle_dataset.json"
    adaptive_dataset = learning_system.create_adaptive_dataset(
        base_dataset,
        "adaptive_motorcycle_dataset.json"
    )
    
    # Generate learning report
    report = learning_system.generate_learning_report()
    
    # Save learning report
    with open("learning_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n📊 Learning Report:")
    print(f"Total interactions: {report['total_interactions']}")
    print(f"Learning patterns: {report['learning_patterns']}")
    print(f"Average confidence: {report['avg_confidence']:.2f}")
    
    print("\n🎯 Adaptive Learning System ready!")
    print("💡 The AI can now learn from chat interactions and improve responses.")

if __name__ == "__main__":
    main()