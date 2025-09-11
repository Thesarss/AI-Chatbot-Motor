from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class ConversationContext:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_topic: Optional[str] = None
        self.user_preferences: Dict[str, Any] = {}
        self.motorcycle_context: Dict[str, Any] = {
            "brand": None,
            "model": None,
            "year": None,
            "engine_type": None,
            "current_issue": None
        }
        self.session_start = datetime.now()
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Menambahkan pesan ke riwayat percakapan"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(message)
        
        # Update topic jika diperlukan
        if role == "user" and metadata:
            if "topic" in metadata:
                self.current_topic = metadata["topic"]
    
    def update_motorcycle_context(self, **kwargs):
        """Update informasi motor pengguna"""
        for key, value in kwargs.items():
            if key in self.motorcycle_context:
                self.motorcycle_context[key] = value
    
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, Any]]:
        """Mendapatkan pesan terbaru"""
        return self.conversation_history[-count:] if self.conversation_history else []
    
    def get_context_summary(self) -> str:
        """Membuat ringkasan konteks untuk prompt"""
        summary_parts = []
        
        # Informasi motor
        if any(self.motorcycle_context.values()):
            motor_info = [f"{k}: {v}" for k, v in self.motorcycle_context.items() if v]
            summary_parts.append(f"Motor: {', '.join(motor_info)}")
        
        # Topic saat ini
        if self.current_topic:
            summary_parts.append(f"Topik: {self.current_topic}")
        
        # Riwayat singkat
        recent_messages = self.get_recent_messages(3)
        if recent_messages:
            history = [f"{msg['role']}: {msg['content'][:50]}..." for msg in recent_messages]
            summary_parts.append(f"Riwayat: {' | '.join(history)}")
        
        return " | ".join(summary_parts)
    
    def save_to_file(self, filepath: str):
        """Menyimpan konteks ke file"""
        data = {
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "current_topic": self.current_topic,
            "user_preferences": self.user_preferences,
            "motorcycle_context": self.motorcycle_context,
            "session_start": self.session_start.isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ConversationContext':
        """Memuat konteks dari file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        context = cls(data["session_id"])
        context.conversation_history = data["conversation_history"]
        context.current_topic = data["current_topic"]
        context.user_preferences = data["user_preferences"]
        context.motorcycle_context = data["motorcycle_context"]
        context.session_start = datetime.fromisoformat(data["session_start"])
        
        return context

class ContextManager:
    def __init__(self):
        self.active_sessions: Dict[str, ConversationContext] = {}
    
    def get_or_create_session(self, session_id: str) -> ConversationContext:
        """Mendapatkan atau membuat session baru"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ConversationContext(session_id)
        return self.active_sessions[session_id]
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Mendapatkan konteks session"""
        return self.active_sessions.get(session_id)
    
    def add_message(self, session_id: str, sender: str, message: str):
        """Menambahkan pesan ke riwayat percakapan"""
        context = self.get_or_create_session(session_id)
        context.add_message(sender, message)
    
    def get_context_summary(self, session_id: str) -> str:
        """Membuat ringkasan konteks percakapan"""
        context = self.get_context(session_id)
        if not context:
            return "Tidak ada riwayat percakapan"
        
        return context.get_context_summary()
    
    def end_session(self, session_id: str, save_path: str = None):
        """Mengakhiri session dan menyimpan jika diperlukan"""
        if session_id in self.active_sessions:
            if save_path:
                self.active_sessions[session_id].save_to_file(save_path)
            del self.active_sessions[session_id]
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Membersihkan session lama"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, context in self.active_sessions.items():
            age = (current_time - context.session_start).total_seconds() / 3600
            if age > max_age_hours:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]