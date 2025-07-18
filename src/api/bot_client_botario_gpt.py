import os
import uuid
import json
import aiohttp
import importlib.util
import logging
import sys
import ssl
import traceback
from api.bot_response import BotResponse

logger = logging.getLogger(__name__)

class BotClient:
    """Client for interacting with the bot API."""
    def __init__(self, config_path=None):
        if config_path:
            config_path = os.path.abspath(config_path)
            spec = importlib.util.spec_from_file_location("bot_config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            self.botario_base_url = getattr(config_module, "BOTARIO_BASE_URL", None)
            self.api_key = getattr(config_module, "BOTARIO_API_TOKEN", None)
            self.iid = getattr(config_module, "BOTARIO_IID", None)
            self.timeout = getattr(config_module, "TIMEOUT", 30)
            self.headers = getattr(config_module, "HEADERS", {})
        else:
            raise ValueError("config_path must be provided for BotClient")

        # Ensure base URL is properly formatted
        if self.botario_base_url and not self.botario_base_url.startswith('http'):
            self.botario_base_url = f"https://{self.botario_base_url}"
        if self.botario_base_url and not self.botario_base_url.endswith('/'):
            self.botario_base_url += '/'

        self.sender_id = str(uuid.uuid4())

        # Create SSL context that allows self-signed certificates
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    async def send_message(self, message: str) -> BotResponse:
        """Send a message to the bot and return a BotResponse."""
        try:
            payload = {
                "payload": {
                    "type": "text",
                    "text": message,
                    "slots": {}
                },
                "sessionId": self.sender_id,
                "startUrl": ""
            }
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': self.api_key
            }
            url = self.botario_base_url.rstrip('/') + f"/api/bots/{self.iid}/chats/send-message"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, ssl=self.ssl_context, timeout=self.timeout) as response:
                    if response.status == 200:
                        try:
                            response_text = await response.text()
                            logger.debug(f"Raw bot response: {response_text}")
                            
                            # Check if response contains multiple JSON objects
                            if response_text.strip().startswith("{") and "}\n{" in response_text:
                                logger.debug(f"Detected multiple JSON objects in response")
                                # Split by closing brace followed by opening brace
                                json_parts = []
                                current_part = ""
                                brace_count = 0
                                
                                for char in response_text:
                                    if char == '{':
                                        brace_count += 1
                                        current_part += char
                                    elif char == '}':
                                        brace_count -= 1
                                        current_part += char
                                        if brace_count == 0:
                                            # End of a JSON object
                                            if current_part.strip():
                                                json_parts.append(current_part.strip())
                                            current_part = ""
                                    else:
                                        if brace_count > 0:  # Only add chars if we're inside a JSON object
                                            current_part += char
                                
                                # Process all found JSON objects
                                texts = []
                                metadata = {}
                                
                                for json_part in json_parts:
                                    try:
                                        data = json.loads(json_part)
                                        bot_text = data.get('payload', {}).get('text', '')
                                        if bot_text:
                                            texts.append(bot_text)
                                        # Use metadata from the first response
                                        if not metadata:
                                            metadata = data.get('payload', {})
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse JSON part: {json_part[:100]}...")
                                
                                if texts:
                                    return BotResponse(text=texts[0], texts=texts, metadata=metadata)
                            
                            # Handle single JSON response or fall back to original parsing
                            try:
                                data = json.loads(response_text)
                            except json.JSONDecodeError as json_err:
                                logger.warning(f"JSON error at position {json_err.pos}, attempting to fix response")
                                
                                # Try multiple strategies to extract valid JSON
                                
                                # Strategy 1: Find first JSON object
                                try:
                                    # Look for the first opening brace and the corresponding closing brace
                                    start = response_text.find('{')
                                    if start >= 0:
                                        # Count braces to find the matching closing brace
                                        brace_count = 0
                                        for i, char in enumerate(response_text[start:], start):
                                            if char == '{':
                                                brace_count += 1
                                            elif char == '}':
                                                brace_count -= 1
                                                if brace_count == 0:
                                                    # Found the matching closing brace
                                                    valid_json_part = response_text[start:i+1]
                                                    data = json.loads(valid_json_part)
                                                    logger.debug(f"Successfully extracted JSON object: {valid_json_part[:100]}...")
                                                    break
                                except Exception as e:
                                    logger.debug(f"Strategy 1 failed: {str(e)}")
                                
                                # Strategy 2: Cut off at error position
                                if 'data' not in locals():
                                    try:
                                        # Try to parse everything up to the error position
                                        valid_json_part = response_text[:json_err.pos].strip()
                                        # Find the last complete JSON object
                                        if valid_json_part.endswith('}'):
                                            data = json.loads(valid_json_part)
                                            logger.debug(f"Successfully parsed truncated JSON: {valid_json_part[:100]}...")
                                    except Exception as e:
                                        logger.debug(f"Strategy 2 failed: {str(e)}")
                                
                                # If all strategies failed, extract text manually
                                if 'data' not in locals():
                                    logger.error(f"All JSON parsing strategies failed, attempting to extract text directly")
                                    # Look for "text" field and extract its value manually
                                    import re
                                    text_match = re.search(r'"text"\s*:\s*"([^"]*)"', response_text)
                                    if text_match:
                                        extracted_text = text_match.group(1)
                                        logger.debug(f"Manually extracted text: {extracted_text}")
                                        return BotResponse(text=extracted_text, metadata={})
                                    else:
                                        logger.error(f"Could not extract text from response")
                                        return BotResponse(text="", metadata={})
                            
                            # Extract text from the response
                            bot_text = data.get('payload', {}).get('text', '')
                            metadata = data.get('payload', {})
                            logger.debug(f"Extracted bot text: {bot_text}")
                            
                            # Check if the bot_text contains multiple responses that should be split
                            # Common patterns in multi-response output
                            texts = []
                            if bot_text:
                                # Some APIs return multiple responses in a single JSON with newlines
                                # If we have multiple paragraphs, treat each as a separate response
                                paragraphs = [p for p in bot_text.split('\n\n') if p.strip()]
                                if len(paragraphs) > 1:
                                    texts = paragraphs
                                else:
                                    texts = [bot_text]
                            
                            return BotResponse(text=bot_text, texts=texts, metadata=metadata)
                        except Exception as e:
                            logger.error(f"Error processing bot response: {str(e)}")
                            logger.debug(f"Exception details: {traceback.format_exc()}")
                            return BotResponse(text="", metadata={})
                    else:
                        logger.error(f"Bot API returned status {response.status}")
                        return BotResponse(text="", metadata={})
        except Exception as e:
            logger.error(f"Exception in send_message: {e}")
            return BotResponse(text="", metadata={})

    def reset_conversation(self) -> None:
        self.sender_id = str(uuid.uuid4())