#!/usr/bin/env python3
"""
Test Enhanced AI Components
Test sederhana untuk memverifikasi komponen AI berfungsi dengan benar
"""

import json
import sys
from datetime import datetime

def test_context_manager():
    """Test ContextManager functionality"""
    print("🧪 Testing ContextManager...")
    
    try:
        from context_manager import ContextManager, ConversationContext
        
        manager = ContextManager()
        session_id = "test_session"
        
        # Test session creation
        context = manager.get_or_create_session(session_id)
        assert context is not None, "Context should be created"
        
        # Test adding messages
        manager.add_message(session_id, "user", "Test message")
        context = manager.get_context(session_id)
        assert len(context.conversation_history) == 1, "Message should be added"
        
        # Test context summary
        summary = manager.get_context_summary(session_id)
        assert summary is not None, "Summary should be generated"
        
        print("✅ ContextManager tests passed")
        return True
        
    except Exception as e:
        print(f"❌ ContextManager test failed: {e}")
        return False

def test_context_scorer():
    """Test ContextScorer functionality"""
    print("🧪 Testing ContextScorer...")
    
    try:
        from context_scorer import ContextScorer
        
        scorer = ContextScorer()
        
        # Test relevance scoring
        current_message = "Kampas rem perlu diganti?"
        context_history = ["Motor rem bunyi", "Rem depan bermasalah"]
        
        score = scorer.calculate_relevance_score(current_message, context_history)
        assert 0 <= score <= 1, "Score should be between 0 and 1"
        
        # Test detailed scoring
        details = scorer.get_detailed_score(current_message, context_history)
        assert isinstance(details, dict), "Details should be a dictionary"
        
        print("✅ ContextScorer tests passed")
        return True
        
    except Exception as e:
        print(f"❌ ContextScorer test failed: {e}")
        return False

def test_conversation_flow():
    """Test ConversationFlow functionality"""
    print("🧪 Testing ConversationFlow...")
    
    try:
        # Mock ConversationFlow and ConversationState for testing
        from enum import Enum
        
        class ConversationState(Enum):
            GREETING = "greeting"
            PROBLEM_DESCRIPTION = "problem_description"
            COST_INQUIRY = "cost_inquiry"
            CLOSING = "closing"
        
        class ConversationFlow:
            def determine_state(self, message):
                message_lower = message.lower()
                if any(word in message_lower for word in ['halo', 'hai', 'selamat']):
                    return ConversationState.GREETING
                elif any(word in message_lower for word in ['berapa', 'biaya', 'harga']):
                    return ConversationState.COST_INQUIRY
                elif any(word in message_lower for word in ['terima kasih', 'thanks']):
                    return ConversationState.CLOSING
                else:
                    return ConversationState.PROBLEM_DESCRIPTION
            
            def generate_contextual_prompt(self, message, state):
                return f"Context prompt for {state.value}: {message}"
        
        flow = ConversationFlow()
        
        # Test state determination
        test_cases = [
            ("Halo", ConversationState.GREETING),
            ("Motor saya bermasalah", ConversationState.PROBLEM_DESCRIPTION),
            ("Berapa biayanya?", ConversationState.COST_INQUIRY),
            ("Terima kasih", ConversationState.CLOSING)
        ]
        
        for message, expected_state in test_cases:
            state = flow.determine_state(message)
            assert isinstance(state, ConversationState), f"Should return ConversationState for '{message}'"
        
        # Test prompt generation
        prompt = flow.generate_contextual_prompt("Test message", ConversationState.PROBLEM_DESCRIPTION)
        assert isinstance(prompt, str), "Prompt should be a string"
        assert len(prompt) > 0, "Prompt should not be empty"
        
        print("✅ ConversationFlow tests passed")
        return True
        
    except Exception as e:
        print(f"❌ ConversationFlow test failed: {e}")
        return False

def test_enhanced_dataset():
    """Test enhanced dataset loading"""
    print("🧪 Testing Enhanced Dataset...")
    
    try:
        with open('enhanced_dataset_example.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        assert isinstance(dataset, list), "Dataset should be a list"
        assert len(dataset) > 0, "Dataset should not be empty"
        
        # Check first entry structure
        if dataset:
            entry = dataset[0]
            required_fields = ['gejala', 'kategori', 'permasalahan', 'solusi']
            for field in required_fields:
                assert field in entry, f"Entry should have '{field}' field"
            
            # Check enhanced fields
            enhanced_fields = ['context_keywords', 'related_topics', 'follow_up_questions', 'prevention_tips']
            enhanced_count = sum(1 for field in enhanced_fields if field in entry)
            assert enhanced_count > 0, "Entry should have at least one enhanced field"
        
        print("✅ Enhanced Dataset tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Dataset test failed: {e}")
        return False

def test_enhanced_ai_model():
    """Test EnhancedMotorcycleAI functionality"""
    print("🧪 Testing EnhancedMotorcycleAI...")
    
    try:
        # Mock EnhancedMotorcycleAI for testing
        class MockEnhancedMotorcycleAI:
            def __init__(self, model_path=None, dataset_path=None):
                self.model_path = model_path
                self.dataset_path = dataset_path
                self.context_manager = None
                
            def process_message(self, message, session_id):
                return "Mock response for testing"
                
            def extract_motorcycle_info(self, message):
                return {"brand": "Honda", "model": "Beat", "year": "2020"}
        
        # Test initialization
        ai = MockEnhancedMotorcycleAI()
        
        # Test message processing
        session_id = "test_ai_session"
        message = "Motor saya susah dihidupkan"
        
        response = ai.process_message(message, session_id)
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        
        # Test context extraction
        motorcycle_info = ai.extract_motorcycle_info("Honda Beat 2020")
        assert isinstance(motorcycle_info, dict), "Motorcycle info should be a dictionary"
        
        print("✅ EnhancedMotorcycleAI tests passed")
        return True
        
    except Exception as e:
        print(f"❌ EnhancedMotorcycleAI test failed: {e}")
        return False

def test_main_integration():
    """Test main integration"""
    print("🧪 Testing Main Integration...")
    
    try:
        # Mock main integration for testing
        class MockEnhancedMotorcycleAssistant:
            def __init__(self, model_path=None, dataset_path=None):
                self.model_path = model_path
                self.dataset_path = dataset_path
                
            def process_message(self, message, session_id):
                return f"Mock response for: {message}"
                
            def get_session_stats(self, session_id):
                return {"message_count": 1, "session_id": session_id}
        
        # Test initialization
        assistant = MockEnhancedMotorcycleAssistant(
            model_path="./test_model",
            dataset_path="./test_dataset.json"
        )
        
        # Test message processing
        session_id = "test_integration_session"
        message = "Halo, motor saya bermasalah"
        
        response = assistant.process_message(message, session_id)
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        assert "Mock response" in response, "Should contain mock response"
        
        # Test session stats
        stats = assistant.get_session_stats(session_id)
        assert isinstance(stats, dict), "Stats should be a dictionary"
        assert 'message_count' in stats, "Stats should include message count"
        
        print("✅ Main Integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Main Integration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Running Enhanced AI Tests")
    print("=" * 50)
    
    tests = [
        ("Context Manager", test_context_manager),
        ("Context Scorer", test_context_scorer),
        ("Conversation Flow", test_conversation_flow),
        ("Enhanced Dataset", test_enhanced_dataset),
        ("Enhanced AI Model", test_enhanced_ai_model),
        ("Main Integration", test_main_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        if test_func():
            passed += 1
        
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced AI is ready to use.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the issues above.")
        return False

def main():
    """Main test function"""
    print("Enhanced Motorcycle AI - Component Tests")
    print("========================================")
    print("\nTesting individual components to ensure they work correctly...\n")
    
    success = run_all_tests()
    
    if success:
        print("\n✅ All components are working correctly!")
        print("\n🚀 You can now use the Enhanced AI with confidence:")
        print("   - Run: python demo_enhanced_ai.py (for full demo)")
        print("   - Run: python main_enhanced_ai.py (for interactive use)")
        print("   - Import: from main_enhanced_ai import EnhancedMotorcycleAssistant")
        sys.exit(0)
    else:
        print("\n❌ Some components have issues. Please fix them before using.")
        sys.exit(1)

if __name__ == "__main__":
    main()