from typing import List, Dict, Any
from pathlib import Path

class ConversationScript:
    def __init__(self, script_path: str):
        """
        Initialize a conversation script from a text file.
        
        Args:
            script_path (str): Path to the conversation script text file
        """
        self.script_path = Path(script_path)
        self.turns = self._load_script()
        self.current_turn = 0
    
    def _load_script(self) -> List[Dict[str, str]]:
        """Load and parse the conversation script."""
        if not self.script_path.exists():
            raise FileNotFoundError(f"Script file not found: {self.script_path}")
        
        with open(self.script_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        turns = []
        current_turn = {}
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:  # Empty line
                continue
            elif line == '---':  # Separator between turns
                if current_turn:
                    turns.append(current_turn)
                    current_turn = {}
                continue
            elif line.startswith('user: '):
                current_turn['user'] = line[6:].strip()
            elif line.startswith('exp: '):
                current_turn['exp'] = line[5:].strip()
        
        # Add the last turn if there is one
        if current_turn:
            turns.append(current_turn)
        
        if not turns:
            raise ValueError("No valid conversation turns found in the script")
        
        return turns
    
    def has_next_turn(self) -> bool:
        """Check if there are more turns in the conversation."""
        return self.current_turn < len(self.turns)
    
    def get_next_turn(self) -> Dict[str, str]:
        """
        Get the next conversation turn.
        
        Returns:
            Dict[str, str]: Dictionary containing user and bot messages
        """
        if not self.has_next_turn():
            raise StopIteration("No more turns in the conversation")
        
        turn = self.turns[self.current_turn]
        self.current_turn += 1
        return turn
    
    def reset(self) -> None:
        """Reset the conversation to the beginning."""
        self.current_turn = 0
        
    @staticmethod
    def extract_bot_text(bot_response: str) -> str:
        """
        Extract the bot's text response from a bot response string.
        
        Args:
            bot_response (str): The bot's response string
            
        Returns:
            str: The extracted text response
        """
        if not bot_response:
            return ""
        return bot_response.strip() 