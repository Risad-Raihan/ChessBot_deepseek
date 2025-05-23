#!/usr/bin/env python3
"""
Chess RAG Chatbot - Main Application

A multi-user, multi-session chess knowledge chatbot using:
- DeepSeek LLM for response generation
- FAISS vector store for RAG retrieval
- JSON-based chat history persistence
"""

import sys
import time
from typing import Dict, List, Optional

from knowledge_base import initialize_knowledge_base
from llm_manager import initialize_llm
from chat_manager import ChatHistoryManager, get_valid_user_id, select_or_create_session


class ChessChatbot:
    """Main chess chatbot class integrating all components."""
    
    def __init__(self):
        self.knowledge_base = None
        self.llm = None
        self.chat_manager = None
        self.current_user_id = None
        self.current_session_id = None
        
        print("🔧 Initializing Chess Chatbot...")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all chatbot components."""
        try:
            # Initialize chat history manager
            print("📚 Loading chat history manager...")
            self.chat_manager = ChatHistoryManager()
            
            # Initialize knowledge base
            print("🧠 Loading chess knowledge base...")
            self.knowledge_base = initialize_knowledge_base()
            
            # Initialize LLM
            print("🤖 Loading language model...")
            self.llm = initialize_llm()
            
            print("✅ All components initialized successfully!\n")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            sys.exit(1)
    
    def setup_user_session(self):
        """Setup user ID and session ID for the conversation."""
        print("=" * 60)
        print("🎯 CHESS CHATBOT - User & Session Setup")
        print("=" * 60)
        
        # Show existing users if any
        existing_users = self.chat_manager.get_all_users()
        if existing_users:
            print("\nExisting users:")
            for i, user_id in enumerate(existing_users, 1):
                sessions = self.chat_manager.get_user_sessions(user_id)
                print(f"  {i}. {user_id} ({len(sessions)} sessions)")
        
        # Get user ID
        self.current_user_id = get_valid_user_id()
        
        # Get session ID
        self.current_session_id = select_or_create_session(self.chat_manager, self.current_user_id)
        
        # Show session info
        summary = self.chat_manager.get_session_summary(self.current_user_id, self.current_session_id)
        print(f"\n✅ Connected to session: {self.current_user_id}/{self.current_session_id}")
        print(f"📊 Session has {summary['message_count']} previous messages")
        
        if summary['message_count'] > 0:
            print("\n📜 Recent conversation history:")
            recent_messages = self.chat_manager.get_recent_messages(
                self.current_user_id, self.current_session_id, limit=4
            )
            for msg in recent_messages:
                role = "You" if msg['role'] == 'user' else "Bot"
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"  {role}: {content}")
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant context from the knowledge base."""
        try:
            results = self.knowledge_base.search(query, top_k=top_k)
            print(f"🔍 Retrieved {len(results)} relevant documents")
            return results
        except Exception as e:
            print(f"⚠️ Error retrieving context: {e}")
            return []
    
    def generate_response(self, user_query: str) -> str:
        """Generate a response using RAG and LLM."""
        try:
            # Retrieve relevant context
            context = self.retrieve_relevant_context(user_query)
            
            # Get conversation history
            history = self.chat_manager.get_recent_messages(
                self.current_user_id, self.current_session_id, limit=10
            )
            
            # Construct prompt
            prompt = self.llm.construct_prompt(history, context, user_query)
            
            # Generate response
            response = self.llm.generate_response(prompt)
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I apologize, but I encountered an error. Please try again."
    
    def process_user_input(self, user_input: str) -> bool:
        """Process user input and return False if user wants to quit."""
        user_input = user_input.strip()
        
        # Check for special commands
        if user_input.lower() in ['quit', 'exit', 'bye']:
            return False
        
        if user_input.lower() in ['help', '/help']:
            self.show_help()
            return True
        
        if user_input.lower() in ['stats', '/stats']:
            self.show_statistics()
            return True
        
        if user_input.lower() in ['history', '/history']:
            self.show_session_history()
            return True
        
        if user_input.lower().startswith('/switch'):
            self.switch_session()
            return True
        
        if user_input.lower().startswith('/export'):
            self.export_current_session()
            return True
        
        # Process as regular chat message
        if not user_input:
            print("Please enter a message or 'help' for commands.")
            return True
        
        # Add user message to history
        self.chat_manager.add_message(
            self.current_user_id, self.current_session_id, 'user', user_input
        )
        
        # Generate and display response
        print("\n🤖 ChessBot is thinking...")
        start_time = time.time()
        
        response = self.generate_response(user_input)
        
        end_time = time.time()
        print(f"⏱️ Response generated in {end_time - start_time:.2f}s")
        
        # Add assistant response to history
        self.chat_manager.add_message(
            self.current_user_id, self.current_session_id, 'assistant', response
        )
        
        # Display response
        print(f"\n🤖 ChessBot: {response}")
        
        return True
    
    def show_help(self):
        """Display help information."""
        print("\n" + "=" * 50)
        print("📖 CHESS CHATBOT HELP")
        print("=" * 50)
        print("Available commands:")
        print("  help, /help     - Show this help message")
        print("  stats, /stats   - Show chat statistics")
        print("  history, /history - Show current session history")
        print("  /switch         - Switch to different user/session")
        print("  /export         - Export current session")
        print("  quit, exit, bye - Exit the chatbot")
        print("\nAsk me anything about:")
        print("  🏆 Chess openings (English, King's Pawn, etc.)")
        print("  📋 Chess rules and regulations")
        print("  📚 Chess history and famous games")
        print("  🎯 Chess strategies and tactics")
        print("=" * 50)
    
    def show_statistics(self):
        """Display chat statistics."""
        stats = self.chat_manager.get_statistics()
        session_summary = self.chat_manager.get_session_summary(
            self.current_user_id, self.current_session_id
        )
        
        print("\n" + "=" * 50)
        print("📊 CHAT STATISTICS")
        print("=" * 50)
        print("Overall Statistics:")
        print(f"  Total Users: {stats['total_users']}")
        print(f"  Total Sessions: {stats['total_sessions']}")
        print(f"  Total Messages: {stats['total_messages']}")
        print(f"  Avg Sessions/User: {stats['average_sessions_per_user']:.1f}")
        print(f"  Avg Messages/Session: {stats['average_messages_per_session']:.1f}")
        
        print(f"\nCurrent Session ({self.current_user_id}/{self.current_session_id}):")
        print(f"  Messages: {session_summary['message_count']}")
        print(f"  User Messages: {session_summary.get('user_messages', 0)}")
        print(f"  Bot Messages: {session_summary.get('assistant_messages', 0)}")
        if session_summary['created']:
            print(f"  Created: {session_summary['created'][:19]}")
        if session_summary['last_activity']:
            print(f"  Last Activity: {session_summary['last_activity'][:19]}")
        print("=" * 50)
    
    def show_session_history(self):
        """Display current session history."""
        history = self.chat_manager.get_session_history(
            self.current_user_id, self.current_session_id
        )
        
        print("\n" + "=" * 50)
        print(f"📜 SESSION HISTORY - {self.current_user_id}/{self.current_session_id}")
        print("=" * 50)
        
        if not history:
            print("No messages in this session yet.")
        else:
            for i, msg in enumerate(history, 1):
                role = "You" if msg['role'] == 'user' else "Bot"
                timestamp = msg['timestamp'][:19]  # Remove microseconds
                print(f"\n[{i}] {role} ({timestamp}):")
                print(f"    {msg['content']}")
        
        print("=" * 50)
    
    def switch_session(self):
        """Switch to a different user or session."""
        print("\n🔄 Switching to different user/session...")
        self.setup_user_session()
    
    def export_current_session(self):
        """Export current session to file."""
        print("\n📤 Exporting current session...")
        try:
            filepath = self.chat_manager.export_session(
                self.current_user_id, self.current_session_id, "json"
            )
            if filepath:
                print(f"✅ Session exported to: {filepath}")
            else:
                print("⚠️ No messages to export in current session.")
        except Exception as e:
            print(f"❌ Error exporting session: {e}")
    
    def run(self):
        """Main chatbot loop."""
        try:
            # Setup user and session
            self.setup_user_session()
            
            # Main chat loop
            print("\n" + "=" * 60)
            print("🎯 CHESS CHATBOT - Ready to chat!")
            print("Type 'help' for commands or 'quit' to exit")
            print("=" * 60)
            
            while True:
                try:
                    # Get user input
                    user_input = input(f"\n[{self.current_user_id}] You: ").strip()
                    
                    # Process input
                    if not self.process_user_input(user_input):
                        print("\n👋 Thanks for using Chess Chatbot! Goodbye!")
                        break
                        
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n\n👋 Goodbye!")
                    break
                    
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            print("Please restart the chatbot.")


def main():
    """Main entry point."""
    print("♟️ Starting Chess RAG Chatbot...")
    
    try:
        chatbot = ChessChatbot()
        chatbot.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 