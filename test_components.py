#!/usr/bin/env python3
"""
Component Test Script

Tests each component of the Chess RAG Chatbot independently
to ensure everything is working before running the full system.
"""

import sys
import traceback
from typing import Dict, List


def test_knowledge_base():
    """Test the knowledge base component."""
    print("🧠 Testing Knowledge Base...")
    try:
        from knowledge_base import initialize_knowledge_base
        
        # Initialize knowledge base
        kb = initialize_knowledge_base()
        
        # Test search functionality
        test_queries = [
            "What is the English opening?",
            "How do you castle in chess?",
            "Chess history"
        ]
        
        for query in test_queries:
            print(f"\n  Testing query: '{query}'")
            results = kb.search(query, top_k=2)
            print(f"  Found {len(results)} results")
            
            if results:
                best_result = results[0]
                print(f"  Best match (score: {best_result['relevance_score']:.3f}): {best_result['text'][:100]}...")
        
        print("✅ Knowledge Base test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Knowledge Base test failed: {e}")
        traceback.print_exc()
        return False


def test_chat_manager():
    """Test the chat manager component."""
    print("\n📚 Testing Chat Manager...")
    try:
        from chat_manager import ChatHistoryManager
        
        # Initialize chat manager
        chat_manager = ChatHistoryManager()
        
        # Test user and session management
        test_user = "test_user_component"
        test_session = "test_session_component"
        
        # Add test messages
        chat_manager.add_message(test_user, test_session, "user", "What is chess?")
        chat_manager.add_message(test_user, test_session, "assistant", "Chess is a strategic board game.")
        chat_manager.add_message(test_user, test_session, "user", "How do I get better at it?")
        
        # Test retrieval
        history = chat_manager.get_session_history(test_user, test_session)
        print(f"  Created session with {len(history)} messages")
        
        # Test statistics
        stats = chat_manager.get_statistics()
        print(f"  System stats: {stats['total_users']} users, {stats['total_messages']} messages")
        
        # Clean up test data
        chat_manager.delete_session(test_user, test_session)
        
        print("✅ Chat Manager test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Chat Manager test failed: {e}")
        traceback.print_exc()
        return False


def test_llm_manager():
    """Test the LLM manager component."""
    print("\n🤖 Testing LLM Manager...")
    try:
        from llm_manager import LLMManager
        
        # Initialize LLM (this will take a while on first run)
        print("  Loading model (this may take a few minutes)...")
        llm = LLMManager()
        llm.load_model()
        
        # Test prompt construction
        test_history = [
            {"role": "user", "content": "What is chess?"},
            {"role": "assistant", "content": "Chess is a strategic board game."}
        ]
        
        test_context = [
            {"text": "Chess is played on an 8x8 board with 64 squares.", "source": "test"}
        ]
        
        test_query = "How do you win at chess?"
        
        # Test prompt construction
        prompt = llm.construct_prompt(test_history, test_context, test_query)
        print(f"  Generated prompt length: {len(prompt)} characters")
        
        # Test response generation (shorter for testing)
        print("  Generating test response...")
        response = llm.generate_response(prompt, max_length=50)
        print(f"  Generated response: {response[:100]}...")
        
        print("✅ LLM Manager test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LLM Manager test failed: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test basic integration between components."""
    print("\n🔗 Testing Component Integration...")
    try:
        from knowledge_base import initialize_knowledge_base
        from llm_manager import initialize_llm
        from chat_manager import ChatHistoryManager
        
        # Initialize all components
        print("  Initializing all components...")
        kb = initialize_knowledge_base()
        llm = initialize_llm()
        chat_manager = ChatHistoryManager()
        
        # Test a simple RAG flow
        test_user = "integration_test"
        test_session = "integration_session"
        test_query = "What is the English opening in chess?"
        
        print(f"  Testing query: '{test_query}'")
        
        # Retrieve context
        context = kb.search(test_query, top_k=2)
        print(f"  Retrieved {len(context)} context documents")
        
        # Get chat history (should be empty for new session)
        history = chat_manager.get_session_history(test_user, test_session)
        print(f"  Chat history length: {len(history)}")
        
        # Generate response
        prompt = llm.construct_prompt(history, context, test_query)
        response = llm.generate_response(prompt, max_length=100)
        print(f"  Generated response: {response[:100]}...")
        
        # Save to chat history
        chat_manager.add_message(test_user, test_session, "user", test_query)
        chat_manager.add_message(test_user, test_session, "assistant", response)
        
        # Verify save
        updated_history = chat_manager.get_session_history(test_user, test_session)
        print(f"  Updated chat history length: {len(updated_history)}")
        
        # Clean up
        chat_manager.delete_session(test_user, test_session)
        
        print("✅ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all component tests."""
    print("🔧 Chess RAG Chatbot - Component Tests")
    print("=" * 50)
    
    tests = [
        ("Knowledge Base", test_knowledge_base),
        ("Chat Manager", test_chat_manager),
        ("LLM Manager", test_llm_manager),
        ("Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print(f"\n⚠️ Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error in {test_name} test: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your chatbot is ready to run.")
        print("Run 'python chess_chatbot.py' to start the full application.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("You may need to install dependencies or check your setup.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
        sys.exit(1) 