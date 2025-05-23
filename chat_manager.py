import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ChatMessage:
    """Represents a single chat message."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    

class ChatHistoryManager:
    """Manages chat history for multiple users and sessions using JSON storage."""
    
    def __init__(self, storage_dir: str = "chat_history"):
        self.storage_dir = storage_dir
        self.chat_data = {}  # In-memory storage: {user_id: {session_id: [messages]}}
        self.storage_file = os.path.join(storage_dir, "chat_history.json")
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing chat history
        self.load_chat_history()
    
    def load_chat_history(self):
        """Load chat history from JSON file."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chat_data = data
                print(f"Loaded chat history from {self.storage_file}")
            except Exception as e:
                print(f"Error loading chat history: {e}")
                self.chat_data = {}
        else:
            print("No existing chat history found. Starting fresh.")
            self.chat_data = {}
    
    def save_chat_history(self):
        """Save current chat history to JSON file."""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_data, f, indent=2, ensure_ascii=False)
            print(f"Chat history saved to {self.storage_file}")
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a specific user."""
        if user_id in self.chat_data:
            return list(self.chat_data[user_id].keys())
        return []
    
    def get_all_users(self) -> List[str]:
        """Get all user IDs."""
        return list(self.chat_data.keys())
    
    def create_session(self, user_id: str, session_id: str) -> bool:
        """Create a new chat session for a user."""
        if user_id not in self.chat_data:
            self.chat_data[user_id] = {}
        
        if session_id not in self.chat_data[user_id]:
            self.chat_data[user_id][session_id] = []
            self.save_chat_history()
            return True
        return False
    
    def add_message(self, user_id: str, session_id: str, role: str, content: str):
        """Add a message to a specific user's session."""
        # Ensure user and session exist
        if user_id not in self.chat_data:
            self.chat_data[user_id] = {}
        if session_id not in self.chat_data[user_id]:
            self.chat_data[user_id][session_id] = []
        
        # Create message
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to chat history
        self.chat_data[user_id][session_id].append(message)
        
        # Save to file
        self.save_chat_history()
    
    def get_session_history(self, user_id: str, session_id: str) -> List[Dict]:
        """Get chat history for a specific user session."""
        if user_id in self.chat_data and session_id in self.chat_data[user_id]:
            return self.chat_data[user_id][session_id]
        return []
    
    def get_recent_messages(self, user_id: str, session_id: str, limit: int = 10) -> List[Dict]:
        """Get the most recent messages from a session."""
        history = self.get_session_history(user_id, session_id)
        return history[-limit:] if history else []
    
    def delete_session(self, user_id: str, session_id: str) -> bool:
        """Delete a specific session."""
        if user_id in self.chat_data and session_id in self.chat_data[user_id]:
            del self.chat_data[user_id][session_id]
            # If user has no sessions left, remove user
            if not self.chat_data[user_id]:
                del self.chat_data[user_id]
            self.save_chat_history()
            return True
        return False
    
    def get_session_summary(self, user_id: str, session_id: str) -> Dict:
        """Get summary information about a session."""
        history = self.get_session_history(user_id, session_id)
        if not history:
            return {"message_count": 0, "created": None, "last_activity": None}
        
        return {
            "message_count": len(history),
            "created": history[0]["timestamp"] if history else None,
            "last_activity": history[-1]["timestamp"] if history else None,
            "user_messages": len([msg for msg in history if msg["role"] == "user"]),
            "assistant_messages": len([msg for msg in history if msg["role"] == "assistant"])
        }
    
    def search_messages(self, user_id: str, session_id: str, query: str) -> List[Dict]:
        """Search for messages containing specific text."""
        history = self.get_session_history(user_id, session_id)
        query_lower = query.lower()
        
        matching_messages = []
        for i, message in enumerate(history):
            if query_lower in message["content"].lower():
                message_with_index = message.copy()
                message_with_index["message_index"] = i
                matching_messages.append(message_with_index)
        
        return matching_messages
    
    def export_session(self, user_id: str, session_id: str, format_type: str = "json") -> str:
        """Export a session to a file."""
        history = self.get_session_history(user_id, session_id)
        if not history:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type.lower() == "json":
            filename = f"export_{user_id}_{session_id}_{timestamp}.json"
            filepath = os.path.join(self.storage_dir, filename)
            
            export_data = {
                "user_id": user_id,
                "session_id": session_id,
                "exported_at": datetime.now().isoformat(),
                "message_count": len(history),
                "messages": history
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
        elif format_type.lower() == "txt":
            filename = f"export_{user_id}_{session_id}_{timestamp}.txt"
            filepath = os.path.join(self.storage_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Chat Export - User: {user_id}, Session: {session_id}\n")
                f.write(f"Exported: {datetime.now().isoformat()}\n")
                f.write("="*50 + "\n\n")
                
                for msg in history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    f.write(f"[{msg['timestamp']}] {role}: {msg['content']}\n\n")
        
        return filepath
    
    def get_statistics(self) -> Dict:
        """Get overall statistics about the chat system."""
        total_users = len(self.chat_data)
        total_sessions = sum(len(sessions) for sessions in self.chat_data.values())
        total_messages = sum(
            len(messages) 
            for user_sessions in self.chat_data.values() 
            for messages in user_sessions.values()
        )
        
        return {
            "total_users": total_users,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "average_sessions_per_user": total_sessions / total_users if total_users > 0 else 0,
            "average_messages_per_session": total_messages / total_sessions if total_sessions > 0 else 0
        }


def get_valid_user_id() -> str:
    """Get a valid user ID from user input."""
    while True:
        user_id = input("\nEnter User ID (alphanumeric, no spaces): ").strip()
        if user_id and user_id.replace("_", "").replace("-", "").isalnum():
            return user_id
        print("Invalid User ID. Please use only letters, numbers, underscores, and hyphens.")


def get_valid_session_id() -> str:
    """Get a valid session ID from user input."""
    while True:
        session_id = input("Enter Session ID (alphanumeric, no spaces): ").strip()
        if session_id and session_id.replace("_", "").replace("-", "").isalnum():
            return session_id
        print("Invalid Session ID. Please use only letters, numbers, underscores, and hyphens.")


def select_or_create_session(chat_manager: ChatHistoryManager, user_id: str) -> str:
    """Helper function to select existing session or create new one."""
    existing_sessions = chat_manager.get_user_sessions(user_id)
    
    if existing_sessions:
        print(f"\nExisting sessions for user '{user_id}':")
        for i, session_id in enumerate(existing_sessions, 1):
            summary = chat_manager.get_session_summary(user_id, session_id)
            print(f"  {i}. {session_id} ({summary['message_count']} messages, last: {summary['last_activity'][:19] if summary['last_activity'] else 'N/A'})")
        
        print(f"  {len(existing_sessions) + 1}. Create new session")
        
        while True:
            try:
                choice = input(f"\nSelect session (1-{len(existing_sessions) + 1}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(existing_sessions):
                    return existing_sessions[choice_num - 1]
                elif choice_num == len(existing_sessions) + 1:
                    return get_valid_session_id()
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        print(f"\nNo existing sessions found for user '{user_id}'.")
        return get_valid_session_id()


if __name__ == "__main__":
    # Test the chat history manager
    chat_manager = ChatHistoryManager()
    
    # Test basic functionality
    test_user = "test_user"
    test_session = "session_1"
    
    # Add some test messages
    chat_manager.add_message(test_user, test_session, "user", "What is the English opening?")
    chat_manager.add_message(test_user, test_session, "assistant", "The English Opening is a chess opening that begins with 1.c4.")
    chat_manager.add_message(test_user, test_session, "user", "How do I play it effectively?")
    
    # Display session history
    history = chat_manager.get_session_history(test_user, test_session)
    print("Test session history:")
    for msg in history:
        role = msg["role"].capitalize()
        print(f"{role}: {msg['content']}")
    
    # Show statistics
    stats = chat_manager.get_statistics()
    print(f"\nStatistics: {stats}") 