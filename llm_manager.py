import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional
import re


class LLMManager:
    """Manages DeepSeek model loading and inference for chess conversations."""
    
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model (smaller, faster model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            self.model.to(self.device)
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try an even smaller fallback
            print("Trying smallest fallback model...")
            self.model_name = "distilgpt2"
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if primary fails."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            print(f"Fallback model {self.model_name} loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load any model: {e}")
    
    def construct_prompt(self, 
                        chat_history: List[Dict], 
                        retrieved_context: List[Dict], 
                        user_query: str) -> str:
        """Construct a well-engineered prompt for chess conversations."""
        
        # System prompt for chess expertise and concise responses
        system_prompt = """You are a knowledgeable chess assistant. Answer questions about chess openings, strategies, rules, and history based on the provided context and conversation history. 

IMPORTANT: Your responses must be:
- Concise and direct
- Focused only on answering the question
- Based on the provided context when relevant
- No reasoning or explanation of your thought process
- Just the final answer

Context from chess knowledge base:"""
        
        # Add retrieved context
        context_section = ""
        if retrieved_context:
            for i, doc in enumerate(retrieved_context[:3]):  # Top 3 results
                context_section += f"\n[Context {i+1}]: {doc['text']}"
        
        # Add conversation history (last 6 turns for context)
        history_section = "\n\nConversation History:"
        if chat_history:
            recent_history = chat_history[-6:]  # Last 3 exchanges (6 messages)
            for msg in recent_history:
                role = "Human" if msg['role'] == 'user' else "Assistant"
                history_section += f"\n{role}: {msg['content']}"
        
        # Current query
        current_query = f"\n\nHuman: {user_query}\nAssistant:"
        
        # Combine all parts
        full_prompt = system_prompt + context_section + history_section + current_query
        
        return full_prompt
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate a response using the loaded model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input with proper parameters
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1500,
                padding=False
            )
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new response (after the prompt)
            response = response[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):]
            
            # Post-process to ensure clean output
            response = self._post_process_response(response)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."
    
    def _post_process_response(self, response: str) -> str:
        """Clean up the model response to ensure concise, direct answers."""
        
        # Remove common prefixes that models often add
        prefixes_to_remove = [
            "I think", "I believe", "In my opinion", "Based on the context",
            "According to the information", "The answer is", "To answer your question",
            "Let me explain", "Here's what I know", "From what I understand"
        ]
        
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
                if response.startswith(",") or response.startswith(":"):
                    response = response[1:].strip()
        
        # Remove reasoning markers
        reasoning_patterns = [
            r"^(Well,?|So,?|Now,?|Actually,?)\s+",
            r"^(Therefore|Thus|Hence),?\s+",
            r"^Let me (think|see|explain).*?[.!?]\s*",
        ]
        
        for pattern in reasoning_patterns:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE)
        
        # Split on sentences and take the first substantive one
        sentences = response.split('. ')
        if sentences:
            # Find first sentence that's substantive (not just filler)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and not any(filler in sentence.lower() 
                    for filler in ['let me', 'i think', 'well', 'actually']):
                    response = sentence
                    break
        
        # Ensure it ends properly
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response


def initialize_llm():
    """Initialize the LLM manager."""
    llm = LLMManager()
    llm.load_model()
    return llm


if __name__ == "__main__":
    # Test the LLM
    llm = initialize_llm()
    
    # Test basic inference
    test_history = [
        {"role": "user", "content": "What is chess?"},
        {"role": "assistant", "content": "Chess is a strategic board game between two players."}
    ]
    
    test_context = [
        {"text": "Chess is a board game played between two players on a checkered board with 64 squares.", "source": "chess_rules.txt"}
    ]
    
    test_query = "How do you win in chess?"
    
    prompt = llm.construct_prompt(test_history, test_context, test_query)
    print("Generated prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    response = llm.generate_response(prompt)
    print(f"Model response: {response}") 