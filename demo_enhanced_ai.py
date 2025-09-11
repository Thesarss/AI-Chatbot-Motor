#!/usr/bin/env python3
"""
Demo Enhanced AI Motorcycle Assistant
Mendemonstrasikan penggunaan AI dengan kemampuan konteks yang ditingkatkan
"""

import json
import time
from datetime import datetime
from enhanced_ai_model import EnhancedMotorcycleAI
from conversation_flow import ConversationFlow, ConversationState
from context_scorer import ContextScorer
from context_manager import ContextManager

def load_enhanced_dataset():
    """Load enhanced dataset untuk demo"""
    try:
        with open('enhanced_dataset_example.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: enhanced_dataset_example.json tidak ditemukan")
        return []

def print_separator(title=""):
    """Print separator untuk output yang lebih rapi"""
    print("\n" + "="*60)
    if title:
        print(f" {title} ")
        print("="*60)
    print()

def demo_basic_conversation():
    """Demo percakapan dasar dengan AI"""
    print_separator("DEMO 1: Percakapan Dasar")
    
    try:
        # Initialize AI dengan model path default
        ai = EnhancedMotorcycleAI(model_path="./bengkelAI-Gemma-LORA-final")
        print("✅ AI initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AI: {str(e)}")
        return
    
    # Simulasi percakapan
    messages = [
        "Halo, motor saya susah dihidupkan",
        "Motor Honda Beat 2020",
        "Sudah 3 hari seperti ini",
        "Belum pernah servis rutin"
    ]
    
    session_id = "demo_session_1"
    
    for i, message in enumerate(messages, 1):
        print(f"User: {message}")
        response = ai.process_message(message, session_id)
        print(f"AI: {response}")
        print(f"\n--- Pesan ke-{i} ---\n")
        time.sleep(1)  # Simulasi delay

def demo_context_awareness():
    """Demo kemampuan AI memahami konteks"""
    print_separator("DEMO 2: Context Awareness")
    
    try:
        # Initialize AI dengan model path default dan terminal ID
        ai = EnhancedMotorcycleAI(
            model_path="./bengkelAI-Gemma-LORA-final",
            terminal_id="885-888"
        )
        print("✅ AI initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AI: {str(e)}")
        return
        
    session_id = "demo_session_2"
    
    # Percakapan dengan topik yang berubah-ubah
    conversation = [
        ("Motor saya bermasalah dengan rem", "Memulai topik rem"),
        ("Rem depan bunyi mencicit", "Melanjutkan topik rem"),
        ("Berapa biaya ganti kampas rem?", "Masih dalam konteks rem"),
        ("Oli mesin kapan harus diganti?", "Pindah topik ke oli"),
        ("Kembali ke masalah rem tadi", "Kembali ke konteks sebelumnya"),
        ("Apakah kampas rem bisa diperbaiki?", "Melanjutkan konteks rem")
    ]
    
    for message, description in conversation:
        print(f"\n[{description}]")
        print(f"User: {message}")
        response = ai.process_message(message, session_id)
        print(f"AI: {response}")
        
        # Tampilkan informasi konteks
        context = ai.context_manager.get_context(session_id)
        if context:
            print(f"\n📊 Context Info:")
            print(f"   - Current Topic: {context.current_topic}")
            print(f"   - Motorcycle Info: {context.motorcycle_info}")
            print(f"   - Message Count: {len(context.conversation_history)}")
        
        time.sleep(1)

def demo_conversation_flow():
    """Demo conversation flow management"""
    print_separator("DEMO 3: Conversation Flow")
    
    flow = ConversationFlow()
    
    # Simulasi berbagai state percakapan
    test_inputs = [
        ("Halo", "Greeting"),
        ("Motor saya bermasalah", "Problem Description"),
        ("Rem bunyi mencicit", "Symptom Details"),
        ("Berapa biayanya?", "Cost Inquiry"),
        ("Terima kasih", "Closing")
    ]
    
    for user_input, expected_state in test_inputs:
        state = flow.determine_state(user_input)
        prompt = flow.generate_contextual_prompt(user_input, state)
        
        print(f"User: {user_input}")
        print(f"Detected State: {state.value}")
        print(f"Expected: {expected_state}")
        print(f"Generated Prompt: {prompt[:100]}...")
        print()
        
        flow.current_state = state
        time.sleep(0.5)

def demo_context_scoring():
    """Demo context scoring system"""
    print_separator("DEMO 4: Context Scoring")
    
    scorer = ContextScorer()
    
    # Test context relevance
    test_cases = [
        {
            "current_message": "Kampas rem perlu diganti?",
            "context_history": ["Motor saya bermasalah dengan rem", "Rem depan bunyi mencicit"],
            "expected": "High relevance (same topic)"
        },
        {
            "current_message": "Berapa harga oli mesin?",
            "context_history": ["Motor saya bermasalah dengan rem", "Rem depan bunyi mencicit"],
            "expected": "Low relevance (different topic)"
        },
        {
            "current_message": "Terima kasih atas sarannya",
            "context_history": ["Coba bersihkan kampas rem", "Periksa juga minyak rem"],
            "expected": "Medium relevance (follow-up)"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {case['expected']}")
        print(f"Current: {case['current_message']}")
        print(f"History: {case['context_history']}")
        
        score = scorer.calculate_relevance_score(
            case['current_message'],
            case['context_history']
        )
        
        print(f"Relevance Score: {score:.2f}")
        
        # Get detailed scoring
        details = scorer.get_detailed_score(
            case['current_message'],
            case['context_history']
        )
        
        print("Detailed Scores:")
        for key, value in details.items():
            print(f"  - {key}: {value:.2f}")
        
        print()
        time.sleep(1)

def demo_session_management():
    """Demo session management capabilities"""
    print_separator("DEMO 5: Session Management")
    
    manager = ContextManager()
    
    # Create multiple sessions
    sessions = [
        ("user_1", ["Motor Yamaha NMAX bermasalah", "Mesin sering mati mendadak"]),
        ("user_2", ["Honda Beat rem blong", "Kampas rem sudah tipis"]),
        ("user_3", ["Kawasaki Ninja oli bocor", "Ada noda oli di lantai"])
    ]
    
    # Add messages to sessions
    for session_id, messages in sessions:
        print(f"\n📱 Session: {session_id}")
        for message in messages:
            manager.add_message(session_id, "user", message)
            print(f"  User: {message}")
        
        # Get context summary
        context = manager.get_context(session_id)
        if context:
            summary = manager.get_context_summary(session_id)
            print(f"  Summary: {summary}")
    
    # Show all active sessions
    print(f"\n📊 Total Active Sessions: {len(manager.active_sessions)}")
    
    # Save and load sessions
    print("\n💾 Saving sessions...")
    try:
        manager.save_session("user_1", "session_user_1.json")
    except Exception as e:
        print(f"Save session error: {e}")
    
    print("✅ Session saved successfully")
    
    # Clean old sessions (demo)
    print("\n🧹 Cleaning old sessions...")
    cleaned = manager.cleanup_old_sessions(max_age_hours=0.001)  # Very short for demo
    print(f"Cleaned {cleaned} old sessions")

def demo_enhanced_dataset():
    """Demo enhanced dataset features"""
    print_separator("DEMO 6: Enhanced Dataset")
    
    dataset = load_enhanced_dataset()
    
    if not dataset:
        print("No enhanced dataset available for demo")
        return
    
    print(f"📚 Loaded {len(dataset)} enhanced entries")
    
    # Show sample enhanced entry
    if dataset:
        sample = dataset[0]
        print("\n📋 Sample Enhanced Entry:")
        print(f"Gejala: {sample['gejala']}")
        print(f"Kategori: {sample['kategori']}")
        print(f"Keywords: {', '.join(sample.get('context_keywords', []))}")
        print(f"Related Topics: {', '.join(sample.get('related_topics', []))}")
        
        if 'follow_up_questions' in sample:
            print("\nFollow-up Questions:")
            for q in sample['follow_up_questions'][:3]:  # Show first 3
                print(f"  - {q}")
        
        if 'prevention_tips' in sample:
            print("\nPrevention Tips:")
            for tip in sample['prevention_tips'][:2]:  # Show first 2
                print(f"  - {tip}")

def demo_full_integration():
    """Demo full integration of all components"""
    print_separator("DEMO 7: Full Integration")
    
    # Initialize main enhanced AI
    try:
        from main_enhanced_ai import EnhancedMotorcycleAssistant
        assistant = EnhancedMotorcycleAssistant(
            model_path="./bengkelAI-Gemma-LORA-final"
        )
        print("🚀 Enhanced Motorcycle Assistant initialized")
        
        # Simulate a complete conversation
        session_id = "full_demo_session"
        
        conversation_flow = [
            "Halo, saya butuh bantuan dengan motor saya",
            "Motor Honda Vario 150, tahun 2021",
            "Masalahnya rem depan bunyi mencicit saat dipakai",
            "Sudah terjadi sekitar 2 minggu",
            "Belum pernah ganti kampas rem",
            "Kira-kira berapa biaya untuk perbaikan?",
            "Apakah bisa diperbaiki sendiri?",
            "Terima kasih atas bantuannya"
        ]
        
        for i, message in enumerate(conversation_flow, 1):
            print(f"\n--- Turn {i} ---")
            print(f"👤 User: {message}")
            
            response = assistant.process_message(message, session_id)
            print(f"🤖 Assistant: {response}")
            
            # Show session stats
            context = assistant.context_manager.get_or_create_session(session_id)
            print(f"\n📊 Session Stats:")
            print(f"   Messages: {len(context.conversation_history)}")
            print(f"   Current Topic: {getattr(context, 'current_topic', 'Unknown')}")
            print(f"   Motorcycle Info: {context.motorcycle_context}")
            print(f"   Context Summary: {context.get_context_summary()[:100]}...")
            
            time.sleep(1.5)
        
        # Generate final summary
        print("\n" + "="*60)
        print(" CONVERSATION SUMMARY ")
        print("="*60)
        
        context = assistant.context_manager.get_or_create_session(session_id)
        summary = context.get_context_summary()
        print(summary)
        
    except ImportError:
        print("❌ Main enhanced AI not available. Please check main_enhanced_ai.py")

def main():
    """Main demo function"""
    print("🔧 Enhanced Motorcycle AI Assistant Demo")
    print("==========================================")
    print("\nDemo ini menunjukkan kemampuan AI dengan context awareness yang ditingkatkan")
    print("untuk memahami percakapan tentang masalah motor secara lebih baik.")
    
    demos = [
        ("1", "Basic Conversation", demo_basic_conversation),
        ("2", "Context Awareness", demo_context_awareness),
        ("3", "Conversation Flow", demo_conversation_flow),
        ("4", "Context Scoring", demo_context_scoring),
        ("5", "Session Management", demo_session_management),
        ("6", "Enhanced Dataset", demo_enhanced_dataset),
        ("7", "Full Integration", demo_full_integration)
    ]
    
    print("\nAvailable Demos:")
    for num, name, _ in demos:
        print(f"  {num}. {name}")
    
    print("\nRunning all demos...\n")
    
    for num, name, demo_func in demos:
        try:
            demo_func()
            print(f"\n✅ Demo {num} ({name}) completed successfully")
        except Exception as e:
            print(f"\n❌ Demo {num} ({name}) failed: {str(e)}")
        
        print("\n" + "-"*60)
        time.sleep(2)
    
    print_separator("DEMO COMPLETED")
    print("🎉 Semua demo telah selesai!")
    print("\nFitur-fitur yang telah didemonstrasikan:")
    print("✓ Context-aware conversation")
    print("✓ Conversation flow management")
    print("✓ Context relevance scoring")
    print("✓ Session management")
    print("✓ Enhanced dataset integration")
    print("✓ Full system integration")
    print("\nAI sekarang siap untuk memberikan bantuan yang lebih kontekstual!")

if __name__ == "__main__":
    main()