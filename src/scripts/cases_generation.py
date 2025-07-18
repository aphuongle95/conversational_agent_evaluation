"""
Script for generating test cases from conversation data.

This script has two modes:

1. Conversation Mode (--mode conversation):
   Generate test cases from existing conversations:
   - From a single conversation/dialog
   - From a JSON file with multiple dialogs

2. Interaction Mode (--mode interaction):
   Update predefined test cases with actual bot responses:
   - Run each test case with the bot
   - Replace expected responses with actual bot responses
   - Support for rate limiting between files

Usage examples:

Conversation Mode:
  python src/scripts/cases_generation.py --mode conversation --conversation-id <ID> --output <OUTPUT_FILE>
  python src/scripts/cases_generation.py --mode conversation --dialog-id <ID> --reseller-token <TOKEN> --output <OUTPUT_FILE>
  python src/scripts/cases_generation.py --mode conversation --json-file <JSON_FILE> [--output-dir <OUTPUT_DIR>]

Interaction Mode:
  python src/scripts/cases_generation.py --mode interaction --test <TEST_FILES> [--output-dir <OUTPUT_DIR>] [--run-wait 30]
"""
import json
import logging
import argparse
import sys
import requests
import asyncio
import glob
import os
import traceback
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

# Import metrics for LLM evaluation (used only in evaluate_with_llm function if needed)
from metrics.llm_metrics import EvaluationMetrics
from api.bot_client_botario_gpt import BotClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration values that will be overridden by config files
BOTARIO_API_TOKEN = None
BOTARIO_BASE_URL = None
BOTARIO_IID = None
BOT_TYPE = "standard"  # Can be "standard" or "gpt" - will be set based on config

def extract_conversation_turns(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract conversation turns from Botario API response.
    
    Combines subsequent user messages and bot messages into single conversation turns.
    All messages from the same speaker are combined into a single message.
    
    Args:
        data: Response from Botario API containing conversation history
        
    Returns:
        List of conversation turns, each with 'user' and 'exp' keys
    """
    turns = []
    current_user_messages = []
    current_bot_messages = []
    
    try:
        # Check if we have the expected data structure
        if not isinstance(data, dict):
            logger.error(f"Invalid API response format: Not a dictionary. Got {type(data)}")
            return []
            
        # Log the response structure for debugging
        logger.debug(f"Response keys: {list(data.keys())}")
            
        if "history" not in data:
            logger.error(f"Invalid API response format: Missing 'history' key. Available keys: {list(data.keys())}")
            return []
        
        messages = data.get("history", [])
        if not isinstance(messages, list):
            logger.error(f"Invalid API response format: 'history' is not a list. Got {type(messages)}")
            return []
        
        # Log the first few messages for debugging
        if len(messages) > 0:
            logger.debug(f"First message structure: {messages[0]}")
            logger.debug(f"Message types found in history: {set(msg.get('role', 'unknown') for msg in messages if isinstance(msg, dict))}")
        
        # Process each message in the history
        for message in messages:
            if not isinstance(message, dict):
                logger.warning(f"Skipping invalid message format (not a dictionary): {message}")
                continue
                
            role = message.get("role", "").lower()
            content = message.get("content", "")
            metadata = message.get("metadata", {})
            
            logger.debug(f"Processing message - role: {role}, content: {content[:50]}{'...' if len(content) > 50 else ''}")
            
            if not content:
                logger.debug(f"Skipping empty content message with role: {role}")
                continue
                
            if role == "user":
                # If we have a previous full turn (user + bot), save it
                if current_user_messages and current_bot_messages:
                    turns.append({
                        "user": "\n".join(current_user_messages),
                        "exp": "\n".join(current_bot_messages)
                    })
                    current_bot_messages = []
                    current_user_messages = []
                
                # Add this user message
                current_user_messages.append(content)
                    
            elif role == "assistant":
                # Only collect bot messages if we already have a user message
                if current_user_messages:
                    current_bot_messages.append(content)
                else:
                    logger.debug("Skipping assistant message because no user message preceded it")
        
        # Save the last turn if exists
        if current_user_messages and current_bot_messages:
            turns.append({
                "user": "\n".join(current_user_messages),
                "exp": "\n".join(current_bot_messages)
            })
        elif current_user_messages:
            logger.warning("Last user message(s) had no bot response")
        
        # Report results
        logger.info(f"Extracted {len(turns)} conversation turns from {len(messages)} messages")
        
        # Special case: If we have messages but no proper turns were found
        if not turns and len(messages) > 0:
            logger.warning("No proper conversation turns found in the history. This may indicate an unexpected response format or an empty conversation.")
            
        return turns
        
    except Exception as e:
        logger.error(f"Error extracting conversation turns: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def save_test_case(turns: List[Dict[str, str]], output_file: str, mode: str = 'conversation'):
    """Save conversation turns as a test case file.
    
    Args:
        turns: List of conversation turns
        output_file: Path to save the test case
        mode: Either 'conversation' or 'interaction' to determine output format
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, turn in enumerate(turns, 1):
            # Always include the user line
            f.write(f"user: {turn['user']}\n")
            
            if mode == 'conversation':
                # In conversation mode, use expected response
                f.write(f"exp: {turn['exp']}\n")
            else:
                # In interaction mode, use actual bot response
                f.write(f"exp: {turn['bot']}\n")
                
            if i < len(turns):
                f.write("---\n")

def load_from_file(file_path: str) -> Dict[str, Any]:
    """Load conversation data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def load_from_api(conversation_id: str) -> Dict[str, Any]:
    """Load conversation data from Botario API.
    
    Args:
        conversation_id: ID of the conversation to fetch
        
    Returns:
        Conversation data as a dictionary
    """
    try:
        # Use different URL formats based on BOT_TYPE
        if BOT_TYPE == "gpt":
            # For Botario GPT API
            url = f"{BOTARIO_BASE_URL.rstrip('/')}/api/bots/{BOTARIO_IID}/chats/{conversation_id}"
            
            # Ensure x-api-key is provided for Botario GPT API
            headers = {
                'accept': 'application/json',
                'x-api-key': BOTARIO_API_TOKEN
            }
        else:
            # For standard Botario API
            url = f"{BOTARIO_BASE_URL.rstrip('/')}/services/chat/conversations/{conversation_id}"
            headers = {
                'accept': 'application/json'
            }
            
        logger.info(f"Fetching conversation data from: {url} (BOT_TYPE: {BOT_TYPE})")
        logger.debug(f"Using headers: {headers}")
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Log response structure to help debug
        if isinstance(data, dict):
            logger.debug(f"Response keys: {list(data.keys())}")
            if "history" in data and isinstance(data["history"], list) and len(data["history"]) > 0:
                logger.debug(f"History contains {len(data['history'])} messages")
                logger.debug(f"First message structure keys: {list(data['history'][0].keys()) if isinstance(data['history'][0], dict) else 'Not a dict'}")
        
        return data
    except Exception as e:
        logger.error(f"Error fetching from Botario API: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def extract_conversation_turns_from_cognitive_voice(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract conversation turns from CognitiveVoice dialog data.
    
    This handles the specific format of the CognitiveVoice dialog API.
    Example format: https://cognitivevoice.io/v1/dialog/{resellerToken}/{dialogId}
    
    The function:
    1. Inserts "/cvg_session" as the first user message
    2. Associates all initial greetings with this first message
    3. Preserves all original user messages and system responses as subsequent turns
    4. Combines consecutive user messages and bot responses into cohesive turns on single lines
    """
    turns = []
    
    # Extract Synthesis and Transcription events
    events = []
    if "data" in data and isinstance(data["data"], list):
        for item in data["data"]:
            if item.get("type") in ["Synthesis", "Transcription"]:
                events.append(item)
    
    # If no events found, return empty list
    if not events:
        return []
    
    # Find the index of the first user transcription
    first_user_index = None
    for i, event in enumerate(events):
        if event.get("type") == "Transcription":
            first_user_index = i
            break
    
    # If no user messages found, return empty list
    if first_user_index is None:
        return []
    
    # Create the special first turn with /cvg_session
    greeting_messages = []
    for i in range(first_user_index):
        if events[i].get("type") == "Synthesis" and "text" in events[i]:
            greeting_messages.append(events[i].get("text", ""))
    
    if greeting_messages:
        turns.append({
            "user": "/cvg_session",
            "exp": " ".join(greeting_messages)
        })
    
    # Process the actual user-bot conversations
    current_user_messages = []
    current_bot_messages = []
    last_type = None
    
    # Start processing from the first real user message
    for i in range(first_user_index, len(events)):
        event = events[i]
        event_type = event.get("type")
        
        if event_type == "Transcription" and "text" in event:
            # If this is a new user message after a bot message sequence, 
            # save the previous exchange first
            if last_type == "Synthesis" and current_user_messages and current_bot_messages:
                turns.append({
                    "user": " ".join(current_user_messages),
                    "exp": " ".join(current_bot_messages)
                })
                current_user_messages = []
                current_bot_messages = []
            
            # Add this user message
            current_user_messages.append(event.get("text", ""))
            last_type = "Transcription"
            
        elif event_type == "Synthesis" and "text" in event and current_user_messages:
            # Add this bot message
            current_bot_messages.append(event.get("text", ""))
            last_type = "Synthesis"
    
    # Save the last exchange if it exists
    if current_user_messages and current_bot_messages:
        turns.append({
            "user": " ".join(current_user_messages),
            "exp": " ".join(current_bot_messages)
        })
    
    return turns

def load_from_cognitive_voice(dialog_id: str, reseller_token: str) -> Dict[str, Any]:
    """Load conversation data from cognitivevoice.io dialog API.
    
    API Reference: https://cognitivevoice.io/specs/?urls.primaryName=Dialog+API#/dialog/getDialog
    URL Structure: /dialog/{resellerToken}/{dialogId}
    
    Example curl:
    curl -X 'GET' \\
      'https://cognitivevoice.io/v1/dialog/reseller_token/dialog_id' \\
      -H 'accept: application/json'
    
    Args:
        dialog_id: Dialog ID for accessing the specific dialog
        reseller_token: Reseller token for authentication (required)
    """
    try:
        # Construct the URL with the required format
        url = f"https://cognitivevoice.io/v1/dialog/{reseller_token}/{dialog_id}"
        logger.info(f"Fetching dialog data from: {url}")
        
        # Header matching the curl example exactly
        headers = {
            'accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Get the data directly from the response
        data = response.json()
        
        # Return the raw data to be processed by the appropriate extraction function
        return data
    except Exception as e:
        logger.error(f"Error fetching from cognitive voice API: {str(e)}")
        raise

async def run_conversation(turns: List[Dict[str, str]], turn_wait: int = 3) -> List[Dict[str, str]]:
    """Run the conversation and capture bot responses.
    
    Args:
        turns: List of conversation turns with user messages and expected responses
        turn_wait: Wait time in seconds between individual turns to avoid rate limits
    """
    # Determine the config file to use
    config_path = os.environ.get("BOT_CONFIG_PATH")
    if config_path:
        config_path_obj = Path(config_path)
        if config_path_obj.is_dir():
            botario_gpt_path = config_path_obj / "botario_gpt.py"
            if botario_gpt_path.exists():
                logger.info(f"Using botario_gpt.py config for BotClient")
                bot_client = BotClient(config_path=str(botario_gpt_path))
            else:
                logger.warning(f"botario_gpt.py not found, falling back to default config")
                bot_client = BotClient(config_path=config_path)
        else:
            # Use the specific file provided
            bot_client = BotClient(config_path=config_path)
    else:
        raise ValueError("BOT_CONFIG_PATH environment variable not set")
        
    conversation_turns = []
    
    for i, turn in enumerate(turns):
        logger.info(f"\nSending message: {turn['user']}")
        
        try:
            # Send message to bot and get response
            response = await bot_client.send_message(turn['user'])
            
            # Check if response is valid
            if not response or not response.content:
                logger.error(f"Invalid response from bot for message: {turn['user']}")
                # Add error turn to conversation and stop
                turn['bot'] = "ERROR: Failed to get valid response from bot"
                conversation_turns.append(turn)
                return conversation_turns
                
            turn['bot'] = response.content
            logger.info(f"Bot response: {response.content}")
            conversation_turns.append(turn)
            
            # Wait between turns to avoid rate limits (except for last turn)
            if i < len(turns) - 1 and turn_wait > 0:
                logger.info(f"Waiting {turn_wait} seconds before next turn...")
                await asyncio.sleep(turn_wait)
            
        except Exception as e:
            logger.error(f"Error during conversation turn: {str(e)}")
            # Add error turn to conversation and stop
            turn['bot'] = f"ERROR: {str(e)}"
            conversation_turns.append(turn)
            return conversation_turns
    
    return conversation_turns

def process_test_files(paths: List[str], output_dir: str):
    """Process test files from paths (files, directories, or patterns).
    
    Args:
        paths: List of paths to process (can be files, directories, or glob patterns)
        output_dir: Directory to save processed files
    """
    try:
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        test_files = []
        
        # Process each path
        for path in paths:
            # Resolve the path to handle relative paths
            path_obj = Path(path).resolve()
            
            # If it's a directory, scan it recursively
            if path_obj.is_dir():
                # Find all .txt files in directory and subdirectories
                dir_files = list(path_obj.rglob("*.txt"))
                if not dir_files:
                    logger.warning(f"No test files found in directory: {path}")
                    continue
                logger.info(f"Found {len(dir_files)} test files in directory: {path}")
                test_files.extend(dir_files)
                continue
            
            # If it's a pattern with wildcards
            if '*' in path or '?' in path:
                matching_files = glob.glob(path)
                if not matching_files:
                    logger.warning(f"No test files found matching pattern: {path}")
                    continue
                logger.info(f"Found {len(matching_files)} test files matching pattern: {path}")
                test_files.extend(matching_files)
                continue
            
            # If it's a single file
            if not path_obj.exists():
                logger.warning(f"Test file not found: {path}")
                continue
            test_files.append(path)
        
        if not test_files:
            logger.error("No valid test files found")
            sys.exit(1)
            
        # Remove duplicates while preserving order
        test_files = list(dict.fromkeys(test_files))
        logger.info(f"Total unique test files to process: {len(test_files)}")
        
        # Process each file
        success_count = 0
        error_count = 0
        
        for i, test_file in enumerate(test_files):
            try:
                logger.info(f"Processing file {test_file} ({i+1}/{len(test_files)})")
                
                # Read the file content
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Process the file (this is just an example - actual processing depends on requirements)
                # In this case, we're simply copying the files to the output directory with the same structure
                relative_path = Path(test_file).relative_to(Path.cwd())
                output_file = output_path / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                logger.info(f"Processed file saved to {output_file}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing file {test_file}: {str(e)}")
                error_count += 1
                
        logger.info(f"File processing completed. Successful: {success_count}, Failed: {error_count}")
        
    except Exception as e:
        logger.error(f"Error in file processing: {str(e)}")
        sys.exit(1)

def process_batch_from_json(json_file_path: str, output_dir: str):
    """Process multiple dialogs from a JSON file.
    
    Expected JSON format:
    {
        "reseller_token": "your_reseller_token_here",
        "dialogs": [
            "dialog_id_1",
            "dialog_id_2",
            "dialog_id_3"
        ]
    }
    """
    try:
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and validate JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or 'reseller_token' not in data or 'dialogs' not in data:
            logger.error(f"Invalid JSON format. Expected a dictionary with 'reseller_token' and 'dialogs' keys.")
            sys.exit(1)
        
        reseller_token = data['reseller_token']
        dialogs = data['dialogs']
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        # Process each dialog
        for i, dialog_id in enumerate(dialogs):
            try:
                logger.info(f"Processing dialog {dialog_id} ({i+1}/{len(dialogs)})")
                
                # Check if output file already exists
                output_file = output_path / f"test_{dialog_id}.txt"
                if output_file.exists():
                    logger.info(f"File {output_file} already exists. Skipping.")
                    skipped_count += 1
                    continue
                
                # Load data
                cognitive_data = load_from_cognitive_voice(
                    dialog_id=dialog_id,
                    reseller_token=reseller_token
                )
                
                # Extract turns
                turns = extract_conversation_turns_from_cognitive_voice(cognitive_data)
                
                if not turns:
                    logger.warning(f"No conversation turns found for dialog {dialog_id}. Skipping.")
                    error_count += 1
                    continue
                
                # Save test case
                save_test_case(turns, str(output_file), mode='conversation')
                logger.info(f"Test case saved to {output_file}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing dialog {dialog_id}: {str(e)}")
                error_count += 1
        
        logger.info(f"Batch processing completed. Successful: {success_count}, Failed: {error_count}, Skipped: {skipped_count}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        sys.exit(1)

async def process_interaction_mode(test_files: List[str], output_dir: str, wait_time: int = 30, turn_wait: int = 3, replace: bool = False):
    """Process test files in interaction mode by running them with the bot.
    
    Args:
        test_files: List of test files to process
        output_dir: Directory to save processed files
        wait_time: Wait time in seconds between files to avoid rate limits
        turn_wait: Wait time in seconds between individual turns
        replace: If True, replace original files instead of creating new ones
    """
    try:
        # Check if Botario config is available
        if not all([BOTARIO_API_TOKEN, BOTARIO_BASE_URL, BOTARIO_IID]):
            logger.error("Botario API configuration is missing or incomplete.")
            logger.error("Please provide a config path with valid Botario API settings.")
            raise ValueError("Missing Botario API configuration")
            
        logger.info("LLM evaluation configuration found and will be used to validate bot responses")
        
        # Convert paths to absolute paths
        output_path = Path(output_dir).resolve()
        if not replace:
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logger.error(f"Permission denied: Cannot create or write to directory '{output_dir}'. Please check your permissions or use a different output directory.")
                sys.exit(1)
        
        # Process each path to get actual test files
        actual_test_files = []
        for path in test_files:
            # Convert to absolute path
            path_obj = Path(path).resolve()
            
            # If it's a directory, scan it recursively
            if path_obj.is_dir():
                # Find all .txt files in directory and subdirectories
                dir_files = list(path_obj.rglob("*.txt"))
                if not dir_files:
                    logger.warning(f"No test files found in directory: {path}")
                    continue
                logger.info(f"Found {len(dir_files)} test files in directory: {path}")
                actual_test_files.extend(dir_files)
                continue
            
            # If it's a pattern with wildcards
            if '*' in path or '?' in path:
                matching_files = glob.glob(path)
                if not matching_files:
                    logger.warning(f"No test files found matching pattern: {path}")
                    continue
                logger.info(f"Found {len(matching_files)} test files matching pattern: {path}")
                actual_test_files.extend(matching_files)
                continue
            
            # If it's a single file
            if not path_obj.exists():
                logger.warning(f"Test file not found: {path}")
                continue
            actual_test_files.append(path_obj)
        
        if not actual_test_files:
            logger.error("No valid test files found")
            sys.exit(1)
            
        # Remove duplicates while preserving order
        actual_test_files = list(dict.fromkeys(actual_test_files))
        logger.info(f"Total unique test files to process: {len(actual_test_files)}")
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        for i, test_file in enumerate(actual_test_files):
            logger.info(f"\nProcessing test file: {test_file} ({i+1}/{len(actual_test_files)})")
            
            try:
                # Determine output file path
                if replace:
                    output_file = test_file
                else:
                    # Create output path maintaining directory structure
                    relative_path = test_file.relative_to(Path.cwd())
                    output_file = output_path / relative_path
                    
                    # Check if output file already exists
                    if output_file.exists():
                        logger.info(f"File {output_file} already exists. Skipping.")
                        skipped_count += 1
                        continue
                
                # Read conversation turns from test file
                turns = read_conversation_script(str(test_file))
                if not turns:
                    logger.error(f"No conversation turns found in {test_file}")
                    error_count += 1
                    continue
                    
                logger.info(f"Loaded {len(turns)} conversation turns from test file")
                
                # Run conversation with bot
                conversation_turns = await run_conversation(turns, turn_wait=turn_wait)
                
                if not conversation_turns:
                    logger.error("No conversation turns were processed. Stopping due to bot connection failure.")
                    error_count += 1
                    continue
                    
                # Check for errors in this run
                has_errors = any('ERROR:' in turn.get('bot', '') for turn in conversation_turns)
                if has_errors:
                    logger.error(f"Run failed due to errors. Skipping file.")
                    error_count += 1
                    continue
                
                # Skip LLM evaluation during case generation
                logger.info("Skipping LLM evaluation for case generation")
                
                # Save updated test case
                try:
                    if not replace:
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                    save_test_case(conversation_turns, str(output_file), mode='interaction')
                    logger.info(f"Updated test case saved to {output_file}")
                    success_count += 1
                except PermissionError:
                    logger.error(f"Permission denied: Cannot write to '{output_file}'. Please check your permissions or use a different output directory.")
                    error_count += 1
                    continue
                
                # Wait between files to avoid rate limits
                if i < len(actual_test_files) - 1:
                    logger.info(f"Waiting {wait_time} seconds before next file...")
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Error processing file {test_file}: {str(e)}")
                error_count += 1
                continue
        
        logger.info(f"\nInteraction mode completed. Successful: {success_count}, Failed: {error_count}, Skipped: {skipped_count}")
        
    except Exception as e:
        logger.error(f"Error in interaction mode: {str(e)}")
        sys.exit(1)

def read_conversation_script(file_path: str) -> List[Dict[str, str]]:
    """Read conversation turns from test file.
    
    Args:
        file_path: Path to the test file
        
    Returns:
        List of conversation turns, each containing 'user' and 'exp' keys
    """
    turns = []
    current_turn = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('user: '):
                if current_turn:
                    turns.append(current_turn)
                current_turn = {'user': line[6:], 'exp': []}
            elif line.startswith('exp: '):
                if current_turn:
                    current_turn['exp'].append(line[5:])
    
    if current_turn:
        turns.append(current_turn)
    
    return turns

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file or directory.
    
    Args:
        config_path: Path to config file or directory
        
    Returns:
        Dictionary containing configuration values
    """
    global BOTARIO_API_TOKEN, BOTARIO_BASE_URL, BOTARIO_IID
    
    try:
        config_path_obj = Path(config_path).resolve()
        config_dir = config_path_obj
        config_file = None
        
        # Set the BOT_CONFIG_PATH environment variable for downstream modules
        os.environ["BOT_CONFIG_PATH"] = str(config_path_obj)
        logger.info(f"Set BOT_CONFIG_PATH environment variable to {config_path_obj}")
        
        # Handle directory vs file path
        if config_path_obj.is_dir():
            # If it's a directory, use base.py by default
            config_file = config_path_obj / "base.py"
            config_dir = config_path_obj
            logger.info(f"Config is a directory, using {config_file}")
        else:
            # If it's a file, use that specific file
            config_file = config_path_obj
            config_dir = config_path_obj.parent
            logger.info(f"Config is a file, using {config_file}")
        
        # Load base configuration
        config = {}
        if config_file.exists():
            # Load Python module
            if config_file.suffix == '.py':
                logger.info(f"Loading Python config from {config_file}")
                spec = importlib.util.spec_from_file_location("bot_config", str(config_file))
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                
                # Extract attributes from the module
                config = {
                    key: value for key, value in vars(config_module).items()
                    if not key.startswith('__')
                }
                
                logger.info(f"Loaded configuration from {config_file}")
            elif config_file.suffix in ['.yml', '.yaml']:
                # Load YAML config
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded YAML configuration from {config_file}")
            else:
                logger.warning(f"Unsupported config file type: {config_file.suffix}. Expected .py, .yml, or .yaml")
        else:
            logger.warning(f"Config file not found: {config_file}")
        
        # Try to load botario_gpt.py first (new preferred config)
        botario_gpt_path = config_dir / "botario_gpt.py"
        if botario_gpt_path.exists():
            logger.info(f"Loading Botario GPT config from {botario_gpt_path}")
            spec = importlib.util.spec_from_file_location("botario_gpt_config", str(botario_gpt_path))
            botario_gpt_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(botario_gpt_module)
            
            # Get Botario values
            BOTARIO_API_TOKEN = getattr(botario_gpt_module, "BOTARIO_API_TOKEN", None)
            BOTARIO_BASE_URL = getattr(botario_gpt_module, "BOTARIO_BASE_URL", None)
            BOTARIO_IID = getattr(botario_gpt_module, "BOTARIO_IID", None)
            # Set bot type to "gpt" when using botario_gpt.py config
            global BOT_TYPE
            BOT_TYPE = "gpt"
            
            if BOTARIO_API_TOKEN and BOTARIO_BASE_URL and BOTARIO_IID:
                logger.info(f"Successfully loaded Botario GPT API configuration (BOT_TYPE={BOT_TYPE})")
            else:
                logger.warning("Incomplete Botario GPT API configuration. Some features may not work.")
        
        # Fall back to botario.py for backward compatibility
        if not (BOTARIO_API_TOKEN and BOTARIO_BASE_URL and BOTARIO_IID):
            botario_config_path = config_dir / "botario.py"
            if botario_config_path.exists():
                logger.info(f"Loading legacy Botario config from {botario_config_path}")
                spec = importlib.util.spec_from_file_location("botario_config", str(botario_config_path))
                botario_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(botario_module)
                
                # Get Botario values
                BOTARIO_API_TOKEN = getattr(botario_module, "BOTARIO_API_TOKEN", BOTARIO_API_TOKEN)
                BOTARIO_BASE_URL = getattr(botario_module, "BOTARIO_BASE_URL", BOTARIO_BASE_URL)
                BOTARIO_IID = getattr(botario_module, "BOTARIO_IID", BOTARIO_IID)
                # Set bot type to "standard" when using legacy botario.py config
                BOT_TYPE = "standard"
                
                if BOTARIO_API_TOKEN and BOTARIO_BASE_URL and BOTARIO_IID:
                    logger.info(f"Successfully loaded legacy Botario API configuration (BOT_TYPE={BOT_TYPE})")
                else:
                    logger.warning("Incomplete legacy Botario API configuration. Some features may not work.")
        
        # Always try to load ai_gateway.py for LLM evaluation
        ai_gateway_path = config_dir / "ai_gateway.py"
        if ai_gateway_path.exists():
            logger.info(f"Loading AI Gateway config from {ai_gateway_path}")
            spec = importlib.util.spec_from_file_location("ai_gateway", str(ai_gateway_path))
            ai_gateway_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ai_gateway_module)
            
            # Get AI Gateway values (no need to assign to globals as they're used via llm_metrics)
            ai_gateway_chat_endpoint = getattr(ai_gateway_module, "AI_GATEWAY_CHAT_ENDPOINT", None)
            ai_gateway_token = getattr(ai_gateway_module, "AI_GATEWAY_TOKEN", None)
            eval_profile = getattr(ai_gateway_module, "EVAL_PROFILE", None)
            
            if ai_gateway_chat_endpoint and ai_gateway_token and eval_profile:
                logger.info("Successfully loaded AI Gateway configuration for LLM evaluation")
            else:
                logger.warning("Incomplete AI Gateway configuration. LLM evaluation may fail.")
            
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        logger.error(traceback.format_exc())
        return {}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate test cases from conversation data')
    
    # Mode selection
    parser.add_argument('--mode', choices=['conversation', 'interaction'], required=True,
                      help='Mode to run in: conversation (from existing data) or interaction (with bot)')
    
    # Input arguments (mutually exclusive for conversation mode)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--conversation-id', help='Botario conversation/chat ID')
    input_group.add_argument('--dialog-id', help='Dialog ID for cognitive voice dialog')
    input_group.add_argument('--json-file', help='JSON file containing a list of dialog IDs and reseller tokens')
    input_group.add_argument('--test', nargs='+', help='Paths to test files or directories (supports glob patterns)')
    
    # Output arguments
    parser.add_argument('--output', help='Output file path for the test case (required for single dialog processing)')
    parser.add_argument('--output-dir', default='cases', help='Output directory for test cases (default: cases)')
    parser.add_argument('--reseller-token', help='Reseller token for authentication with cognitive voice API (required for --dialog-id)')
    
    # Configuration
    parser.add_argument('--config', type=str, required=True, 
                      help='Path to configuration file or directory (if directory, uses botario_gpt.py if available, falls back to base.py)')
    
    # Interaction mode specific arguments
    parser.add_argument('--run-wait', type=int, default=30, help='Wait time in seconds between files to avoid rate limits (default: 30)')
    parser.add_argument('--turn-wait', type=int, default=3, help='Wait time in seconds between individual turns (default: 3)')
    parser.add_argument('--replace', action='store_true', help='Replace original files instead of creating new ones')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase verbosity level (-v for INFO, -vv for DEBUG)')
    
    args = parser.parse_args()
    
    # Set up logging based on verbosity level
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set third-party loggers to WARNING to reduce noise
    for logger_name in ['asyncio', 'aiohttp', 'urllib3', 'httpx']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Validate arguments based on mode
    if args.mode == 'conversation':
        if (args.dialog_id and not args.reseller_token):
            logger.error("The reseller-token parameter is required when using dialog-id")
            sys.exit(1)
        
        if (args.conversation_id or args.dialog_id) and not args.output:
            logger.error("The output parameter is required when processing a single dialog or conversation")
            sys.exit(1)
            
        if args.test:
            logger.error("The --test parameter is only available in interaction mode")
            sys.exit(1)
    
    try:
        # Process config path
        config_path = Path(args.config).resolve()
        if config_path.is_dir():
            # If it's a directory, we'll prioritize botario_gpt.py, then fall back to base.py
            botario_gpt_path = config_path / "botario_gpt.py"
            if botario_gpt_path.exists():
                logger.info(f"Config is a directory, found botario_gpt.py")
                # Set BOT_CONFIG_PATH to the directory
                os.environ["BOT_CONFIG_PATH"] = str(config_path)
            else:
                logger.info(f"Config is a directory, botario_gpt.py not found, using base.py")
                os.environ["BOT_CONFIG_PATH"] = str(config_path)
        else:
            # For a specific file, use that exact file
            os.environ["BOT_CONFIG_PATH"] = str(config_path)
        
        logger.info(f"Set BOT_CONFIG_PATH to {os.environ['BOT_CONFIG_PATH']}")
        
        # Load configuration with the processed path
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        if not config:
            logger.warning("No configuration loaded or empty configuration. Using defaults.")
        
        # Ensure we have required Botario API configuration
        if not all([BOTARIO_API_TOKEN, BOTARIO_BASE_URL, BOTARIO_IID]):
            logger.error("Missing required Botario API configuration. Please check your config file.")
            logger.error(f"API Token: {'Set' if BOTARIO_API_TOKEN else 'Missing'}")
            logger.error(f"Base URL: {'Set' if BOTARIO_BASE_URL else 'Missing'}")
            logger.error(f"IID: {'Set' if BOTARIO_IID else 'Missing'}")
            sys.exit(1)
            
        if args.mode == 'conversation':
            # Handle JSON file input (batch processing)
            if args.json_file:
                process_batch_from_json(args.json_file, args.output_dir)
            # Handle single dialog or conversation
            else:
                if args.conversation_id:
                    logger.info(f"Fetching conversation from Botario API: {args.conversation_id}")
                    conversation_data = load_from_api(args.conversation_id)
                    turns = extract_conversation_turns(conversation_data)
                else:  # args.dialog_id
                    logger.info(f"Fetching dialog from Cognitive Voice API: {args.dialog_id}")
                    cognitive_data = load_from_cognitive_voice(
                        dialog_id=args.dialog_id,
                        reseller_token=args.reseller_token
                    )
                    turns = extract_conversation_turns_from_cognitive_voice(cognitive_data)
                
                if not turns:
                    logger.error("No conversation turns were extracted from the data")
                    logger.error("Please check that the conversation ID is valid and contains messages")
                    sys.exit(1)
                
                logger.info(f"Successfully extracted {len(turns)} conversation turns")
                
                try:
                    # Save test case without LLM evaluation
                    save_test_case(turns, args.output, mode='conversation')
                    logger.info(f"Test case successfully saved to {args.output}")
                except Exception as save_error:
                    logger.error(f"Failed to save test case: {str(save_error)}")
                    logger.error(traceback.format_exc())
                    sys.exit(1)
        
        else:  # interaction mode
            if not args.test:
                logger.error("The --test parameter is required in interaction mode")
                sys.exit(1)
            
            # Process test files in interaction mode
            asyncio.run(process_interaction_mode(
                args.test,
                args.output_dir,
                wait_time=args.run_wait,
                turn_wait=args.turn_wait,
                replace=args.replace
            ))
        
    except Exception as e:
        logger.error(f"Error generating test cases: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()

async def evaluate_with_llm(turns: List[Dict[str, str]]) -> Dict[str, Any]:
    """Evaluate conversation turns using LLM metrics.
    
    Args:
        turns: List of conversation turns to evaluate
        
    Returns:
        Dictionary containing evaluation results
    """
    try:
        # Initialize metrics
        metrics = EvaluationMetrics()
        
        # Process turns to handle multiple bot responses
        processed_turns = []
        for turn in turns:
            processed_turn = {
                'user': turn.get('user', ''),
                'exp': turn.get('exp', '')
            }
            
            # Handle multiple bot responses by joining them with newlines
            if isinstance(turn.get('bot'), list):
                processed_turn['bot'] = '\n'.join(turn.get('bot', []))
            else:
                processed_turn['bot'] = turn.get('bot', '')
            
            processed_turns.append(processed_turn)
        
        # Evaluate conversation
        logger.info(f"Evaluating {len(processed_turns)} turns with LLM")
        evaluation_results = await metrics.evaluate_conversation(processed_turns)
        
        # Add summary statistics
        if 'turn_evaluations' in evaluation_results:
            successful_turns = sum(1 for eval in evaluation_results['turn_evaluations'] if eval['success'])
            total_evaluated = len(evaluation_results['turn_evaluations'])
            avg_score = sum(eval['score'] for eval in evaluation_results['turn_evaluations']) / total_evaluated if total_evaluated > 0 else 0
            success_rate = (successful_turns/total_evaluated)*100 if total_evaluated > 0 else 0
            
            evaluation_results['summary'] = {
                'total_turns_evaluated': total_evaluated,
                'successful_turns': successful_turns,
                'failed_turns': total_evaluated - successful_turns,
                'success_rate': round(success_rate, 1),
                'average_score': round(avg_score, 2)
            }
            
            logger.info(f"LLM Evaluation: {successful_turns}/{total_evaluated} turns successful ({success_rate:.1f}%)")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error during LLM evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        raise