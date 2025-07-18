from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

@dataclass
class BotResponse:
    """Base class for bot responses."""
    text: str = ""
    texts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization to handle different input formats.
        
        This ensures compatibility whether initialized with 'text' or 'texts'
        """
        # If content is provided (from bot_client_botario.py), convert it to text
        if hasattr(self, 'content'):
            self.text = getattr(self, 'content')
            delattr(self, 'content')
            
        # If neither text nor texts was explicitly set, default to empty
        if not self.text and not self.texts:
            self.text = ""
        # If both text and texts are provided, prefer texts
        elif self.text and self.texts:
            pass  # Keep both as is
        # If only text is provided and it's not empty, add it to texts
        elif self.text and not self.texts:
            self.texts = [self.text]
        # If only texts is provided, set text to be the first item or empty
        elif not self.text and self.texts:
            self.text = self.texts[0] if self.texts else ""
    
    @classmethod
    def from_dict(cls, response_dict: Dict[str, Any]) -> 'BotResponse':
        """Create a BotResponse from a dictionary.
        
        Args:
            response_dict: Dictionary containing the response data
            
        Returns:
            A BotResponse instance
        """
        text = response_dict.get('text', '')
        texts = response_dict.get('texts', [])
        metadata = response_dict.get('metadata', {})
        
        # If 'content' is in the dict (from bot_client_botario.py), use it as text
        if 'content' in response_dict:
            text = response_dict.get('content', '')
            
        # Create instance with all available data
        return cls(text=text, texts=texts, metadata=metadata)
    
    def get_responses(self) -> List[str]:
        """Get all bot responses.
        
        Returns:
            List of bot response texts
        """
        # If texts is populated, return it; otherwise return text as a single-item list
        if self.texts:
            return self.texts
        elif self.text:
            return [self.text]
        else:
            return []
