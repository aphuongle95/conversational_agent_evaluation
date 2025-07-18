import os
import uuid
import json
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from config.botario import BOTARIO_URL, BOTARIO_IID, TIMEOUT, HEADERS
from config.ai_gateway import AI_GATEWAY_TOKEN, AI_GATEWAY_CHAT_ENDPOINT, EVAL_PROFILE
import logging
import sys
from dataclasses import dataclass
import aiohttp

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Import the shared BotResponse class 
from api.bot_response import BotResponse

class BotClient:
    """Client for interacting with the Botario API."""
    
    def __init__(self):
        """
        Initialize the bot client.
        """
        self.base_url = BOTARIO_URL
        self.iid = BOTARIO_IID
        self.headers = HEADERS
        self.timeout = TIMEOUT
        # Generate a unique sender ID for this conversation
        self.sender_id = str(uuid.uuid4())
        self.conversation_id = None
        self.api_key = AI_GATEWAY_TOKEN
        self.api_base = AI_GATEWAY_CHAT_ENDPOINT
        self.eval_profile = EVAL_PROFILE
    
    def _extract_text_responses(self, data: List[Dict[str, Any]]) -> str:
        """
        Extract and combine all text responses from Botario API response.
        Handles both direct text responses and custom cvg_call_say responses.
        Safely handles None values in the response structure.
        
        Args:
            data: List of response objects from the API
            
        Returns:
            Combined text from all responses
        """
        text_responses = []
        for msg in data:
            # Skip if message is not a bot event
            if not msg or msg.get("event") != "bot":
                continue
                
            # Check for direct text response first
            if msg.get("text"):
                text_responses.append(msg["text"])
            # Then check for custom cvg_call_say response
            elif msg.get("data", {}).get("custom") and msg["data"]["custom"].get("cvg_call_say", {}).get("text"):
                text_responses.append(msg["data"]["custom"]["cvg_call_say"]["text"])
                
        # Join all responses with spaces
        return " ".join(text_responses) if text_responses else ""
    
    async def send_message(self, message: str) -> BotResponse:
        """
        Send a message to the bot and get the response.
        
        Args:
            message (str): The message to send
            
        Returns:
            BotResponse: The bot's response
        """
        try:
            # Prepare the request payload
            payload = {
                "iid": self.iid,
                "message": message,
                "sender": self.sender_id
            }
            
            if self.conversation_id:
                payload["conversation_id"] = self.conversation_id
            
            logger.debug(f"Sending request to {self.base_url}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"Raw bot response: {json.dumps(data, indent=2)}")
                        
                        if not data:
                            return BotResponse(
                                text="",
                                metadata={"error": "No response received"}
                            )
                        
                        # Extract text from all bot events
                        text = self._extract_text_responses(data)
                        if text:
                            # Update conversation ID if present in any event
                            for event in data:
                                if "conversation_id" in event:
                                    self.conversation_id = event["conversation_id"]
                                    break
                            
                            return BotResponse(
                                text=text,
                                metadata={
                                    "conversation_id": self.conversation_id,
                                    "events": data
                                }
                            )
                        
                        logger.error("No valid bot response found in events")
                        return BotResponse(
                            text="",
                            metadata={"error": "No text responses found in bot reply"}
                        )
                    else:
                        error_text = await response.text()
                        logger.error(f"Error from bot API: {error_text}")
                        return BotResponse(
                            text="",
                            metadata={"error": f"API error: {error_text}"}
                        )
                        
        except Exception as e:
            logger.error(f"Error sending message to bot: {str(e)}", exc_info=True)
            return BotResponse(
                text="",
                metadata={"error": str(e)}
            )
    
    def reset_conversation(self) -> None:
        """Reset the conversation by generating a new sender ID."""
        self.sender_id = str(uuid.uuid4())

    async def send_message_through_api(self, message: str, conversation_id: Optional[str] = None) -> BotResponse:
        """
        Send a message to the bot and get its response through the API gateway.
        
        Args:
            message: The user's message to send
            conversation_id: Optional conversation ID for maintaining context
            
        Returns:
            BotResponse object containing the bot's response and any metadata
        """
        try:
            # Prepare the request payload
            payload = {
                "messages": [{"role": "user", "content": message}],
                "eval_profile": self.eval_profile
            }
            
            if conversation_id:
                payload["conversation_id"] = conversation_id
                
            logger.debug(f"Sending request to {self.api_base}")
            logger.debug(f"Payload: {payload}")
            
            # TODO: Implement actual API call here
            # This is a placeholder for the actual implementation
            # You would typically use aiohttp or httpx for async HTTP requests
            
            # For now, return a mock response
            return BotResponse(
                text="This is a mock response. Implement actual API call.",
                metadata={"conversation_id": conversation_id}
            )
            
        except Exception as e:
            logger.error(f"Error sending message to bot: {str(e)}", exc_info=True)
            raise

    async def start_conversation(self) -> str:
        """
        Start a new conversation with the bot.
        
        Returns:
            conversation_id: The ID of the new conversation
        """
        try:
            # TODO: Implement conversation initialization
            # This would typically involve calling the API to create a new conversation
            # and returning the conversation ID
            
            # For now, return a mock conversation ID
            return "mock_conversation_id"
            
        except Exception as e:
            logger.error(f"Error starting conversation: {str(e)}", exc_info=True)
            raise
            
    async def end_conversation(self, conversation_id: str) -> None:
        """
        End a conversation with the bot.
        
        Args:
            conversation_id: The ID of the conversation to end
        """
        try:
            # TODO: Implement conversation cleanup
            # This would typically involve calling the API to clean up the conversation
            
            logger.debug(f"Ending conversation: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Error ending conversation: {str(e)}", exc_info=True)
            raise 