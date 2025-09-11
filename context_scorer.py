import re
from typing import Dict, List, Any, Tuple
from collections import Counter
import math

class ContextScorer:
    def __init__(self):
        # Bobot untuk berbagai faktor scoring
        self.weights = {
            "keyword_match": 0.3,
            "topic_consistency": 0.25,
            "temporal_relevance": 0.2,
            "semantic_similarity": 0.15,
            "user_intent": 0.1
        }
        
        # Keywords untuk setiap kategori motor
        self.category_keywords = {
            "rem": {
                "primary": ["rem", "brake", "berhenti", "pengereman"],
                "secondary": ["kampas", "cakram", "tromol", "blong", "keras", "bunyi", "squeaking"]
            },
            "mesin": {
                "primary": ["mesin", "engine", "hidup", "mati", "starter"],
                "secondary": ["piston", "silinder", "kompresi", "oli", "overheat", "knocking"]
            },
            "transmisi": {
                "primary": ["transmisi", "gigi", "kopling", "cvt", "matic"],
                "secondary": ["perpindahan", "slip", "keras", "bunyi", "getaran"]
            },
            "kelistrikan": {
                "primary": ["aki", "battery", "listrik", "starter", "lampu"],
                "secondary": ["alternator", "kabel", "sekring", "fuse", "charging"]
            },
            "bahan_bakar": {
                "primary": ["bensin", "fuel", "bahan bakar", "tangki"],
                "secondary": ["karburator", "injeksi", "filter", "pompa", "bocor"]
            },
            "suspensi": {
                "primary": ["shock", "per", "suspensi", "ban"],
                "secondary": ["velg", "bearing", "keras", "empuk", "bocor"]
            }
        }
        
        # Intent patterns
        self.intent_patterns = {
            "diagnosis": [r"kenapa", r"mengapa", r"apa penyebab", r"masalah", r"gejala"],
            "solution": [r"bagaimana", r"cara", r"solusi", r"perbaiki", r"atasi"],
            "cost": [r"berapa", r"biaya", r"harga", r"mahal", r"murah"],
            "prevention": [r"mencegah", r"hindari", r"perawatan", r"maintenance"]
        }
    
    def calculate_keyword_score(self, text: str, category: str) -> float:
        """Menghitung skor berdasarkan kecocokan keyword"""
        if category not in self.category_keywords:
            return 0.0
        
        text_lower = text.lower()
        keywords = self.category_keywords[category]
        
        primary_matches = sum(1 for kw in keywords["primary"] if kw in text_lower)
        secondary_matches = sum(1 for kw in keywords["secondary"] if kw in text_lower)
        
        # Primary keywords memiliki bobot lebih tinggi
        total_score = (primary_matches * 2) + secondary_matches
        max_possible = (len(keywords["primary"]) * 2) + len(keywords["secondary"])
        
        return total_score / max_possible if max_possible > 0 else 0.0
    
    def calculate_topic_consistency(self, current_topic: str, conversation_history: List[Dict[str, Any]]) -> float:
        """Menghitung konsistensi topik dalam percakapan"""
        if not conversation_history or not current_topic:
            return 0.0
        
        # Ambil 5 pesan terakhir
        recent_messages = conversation_history[-5:]
        topic_mentions = 0
        
        for message in recent_messages:
            if message.get("metadata", {}).get("topic") == current_topic:
                topic_mentions += 1
            
            # Cek juga dalam konten pesan
            content = message.get("content", "").lower()
            if any(kw in content for kw in self.category_keywords.get(current_topic, {}).get("primary", [])):
                topic_mentions += 0.5
        
        return min(topic_mentions / len(recent_messages), 1.0)
    
    def calculate_temporal_relevance(self, conversation_history: List[Dict[str, Any]], decay_factor: float = 0.8) -> float:
        """Menghitung relevansi temporal (pesan terbaru lebih relevan)"""
        if not conversation_history:
            return 0.0
        
        total_weight = 0.0
        weighted_relevance = 0.0
        
        for i, message in enumerate(reversed(conversation_history[-10:])):
            weight = decay_factor ** i
            relevance = 1.0  # Asumsi semua pesan relevan, bisa disesuaikan
            
            weighted_relevance += weight * relevance
            total_weight += weight
        
        return weighted_relevance / total_weight if total_weight > 0 else 0.0
    
    def detect_user_intent(self, text: str) -> Tuple[str, float]:
        """Mendeteksi intent user dan confidence score"""
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent] / sum(intent_scores.values())
            return best_intent, confidence
        
        return "unknown", 0.0
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Menghitung similaritas semantik sederhana berdasarkan kata"""
        # Tokenisasi sederhana
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_relevance_score(self, current_message: str, context_history: List[str]) -> float:
        """Calculate overall relevance score"""
        return self.score_context_relevance(current_message, context_history)
    
    def score_context_relevance(self, current_message: str, context_history: List[str]) -> float:
        """Score context relevance between current message and history"""
        if not context_history:
            return 0.0
        
        # Convert to conversation format for compatibility
        conversation_history = []
        for msg in context_history:
            conversation_history.append({"content": msg, "metadata": {}})
        
        scores = self.calculate_context_relevance(current_message, conversation_history)
        return scores.get("total_score", 0.0)
    
    def get_detailed_score(self, current_message: str, context_history: List[str]) -> dict:
        """Get detailed scoring breakdown"""
        if not context_history:
            return {"total_score": 0.0, "breakdown": {}}
        
        # Convert to conversation format
        conversation_history = []
        for msg in context_history:
            conversation_history.append({"content": msg, "metadata": {}})
        
        return self.calculate_context_relevance(current_message, conversation_history)
    
    def calculate_context_relevance(self, 
                                  current_input: str, 
                                  conversation_history: List[Dict[str, Any]], 
                                  current_topic: str = None) -> Dict[str, float]:
        """Menghitung skor relevansi konteks secara keseluruhan"""
        scores = {}
        
        # 1. Keyword matching score
        if current_topic:
            scores["keyword_match"] = self.calculate_keyword_score(current_input, current_topic)
        else:
            # Cari kategori dengan skor tertinggi
            category_scores = {}
            for category in self.category_keywords.keys():
                category_scores[category] = self.calculate_keyword_score(current_input, category)
            
            if category_scores:
                best_category = max(category_scores, key=category_scores.get)
                scores["keyword_match"] = category_scores[best_category]
                current_topic = best_category
            else:
                scores["keyword_match"] = 0.0
        
        # 2. Topic consistency score
        scores["topic_consistency"] = self.calculate_topic_consistency(current_topic, conversation_history)
        
        # 3. Temporal relevance score
        scores["temporal_relevance"] = self.calculate_temporal_relevance(conversation_history)
        
        # 4. Semantic similarity dengan pesan terakhir
        if conversation_history:
            last_message = conversation_history[-1].get("content", "")
            scores["semantic_similarity"] = self.calculate_semantic_similarity(current_input, last_message)
        else:
            scores["semantic_similarity"] = 0.0
        
        # 5. User intent score
        intent, confidence = self.detect_user_intent(current_input)
        scores["user_intent"] = confidence
        
        # Hitung weighted total score
        total_score = sum(scores[factor] * self.weights[factor] for factor in scores)
        scores["total_score"] = total_score
        scores["detected_topic"] = current_topic
        scores["detected_intent"] = intent
        
        return scores
    
    def should_maintain_context(self, scores: Dict[str, float], threshold: float = 0.6) -> bool:
        """Menentukan apakah konteks harus dipertahankan"""
        return scores.get("total_score", 0.0) >= threshold
    
    def get_context_suggestions(self, scores: Dict[str, float]) -> List[str]:
        """Memberikan saran untuk meningkatkan konteks"""
        suggestions = []
        
        if scores.get("keyword_match", 0) < 0.3:
            suggestions.append("Pertanyaan kurang spesifik untuk kategori motor")
        
        if scores.get("topic_consistency", 0) < 0.4:
            suggestions.append("Topik percakapan tidak konsisten")
        
        if scores.get("semantic_similarity", 0) < 0.2:
            suggestions.append("Pertanyaan tidak terkait dengan percakapan sebelumnya")
        
        if scores.get("user_intent", 0) < 0.3:
            suggestions.append("Intent user tidak jelas")
        
        return suggestions

# Contoh penggunaan
if __name__ == "__main__":
    scorer = ContextScorer()
    
    # Simulasi conversation history
    conversation_history = [
        {
            "role": "user",
            "content": "Motor saya remnya blong",
            "metadata": {"topic": "rem"}
        },
        {
            "role": "assistant",
            "content": "Rem blong bisa disebabkan kampas rem habis atau minyak rem kurang",
            "metadata": {"topic": "rem"}
        }
    ]
    
    # Test scoring
    current_input = "Berapa biaya ganti kampas rem?"
    scores = scorer.calculate_context_relevance(current_input, conversation_history, "rem")
    
    print("Context Scores:")
    for key, value in scores.items():
        print(f"{key}: {value:.3f}")
    
    print(f"\nMaintain context: {scorer.should_maintain_context(scores)}")
    
    suggestions = scorer.get_context_suggestions(scores)
    if suggestions:
        print("\nSuggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion}")