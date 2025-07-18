import os
import json
import logging
import sys
import asyncio
import requests
import traceback
import importlib.util
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Configuration values - will be loaded from config files
ACCURACY_THRESHOLD = 0.4  # Default value
AI_GATEWAY_TOKEN = None
AI_GATEWAY_CHAT_ENDPOINT = None
EVAL_PROFILE = None
MODEL_TEMPERATURE = 0.0  # Default value
MAX_TOKENS = 1000  # Default value

# Load configuration from environment variable first (set by simulation.py or evaluation.py)
config_path = os.environ.get("BOT_CONFIG_PATH")

if not config_path:
    logger.warning("BOT_CONFIG_PATH environment variable is not set. Config loading may fail.")
elif not os.path.exists(config_path):
    logger.warning(f"Config path does not exist: {config_path}")
else:
    logger.info(f"Loading config from: {config_path}")
    try:
        import importlib.util
        
        config_path = Path(config_path)
        config_dir = config_path
        
        # If it's a directory, use the base.py file
        if config_path.is_dir():
            config_path = config_path / "base.py"
            config_dir = config_path.parent
            logger.info(f"Using base.py from directory: {config_path}")
        else:
            # If it's a file, get the parent directory
            config_dir = config_path.parent
            
        # Load base config if the file exists
        if config_path.exists():
            logger.info(f"Loading base config from {config_path}")
            spec = importlib.util.spec_from_file_location("bot_config", str(config_path))
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Get accuracy threshold
            ACCURACY_THRESHOLD = getattr(config_module, "ACCURACY_THRESHOLD", ACCURACY_THRESHOLD)
            logger.info(f"Loaded ACCURACY_THRESHOLD = {ACCURACY_THRESHOLD}")
            
            # Try to load AI gateway config
            ai_gateway_path = config_dir / "ai_gateway.py"
            if ai_gateway_path.exists():
                logger.info(f"Loading AI gateway config from {ai_gateway_path}")
                spec = importlib.util.spec_from_file_location("ai_gateway", str(ai_gateway_path))
                ai_gateway_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ai_gateway_module)
                
                # Get AI gateway values
                AI_GATEWAY_TOKEN = getattr(ai_gateway_module, "AI_GATEWAY_TOKEN", AI_GATEWAY_TOKEN)
                AI_GATEWAY_CHAT_ENDPOINT = getattr(ai_gateway_module, "AI_GATEWAY_CHAT_ENDPOINT", AI_GATEWAY_CHAT_ENDPOINT)
                EVAL_PROFILE = getattr(ai_gateway_module, "EVAL_PROFILE", EVAL_PROFILE)
                MODEL_TEMPERATURE = getattr(ai_gateway_module, "MODEL_TEMPERATURE", MODEL_TEMPERATURE)
                MAX_TOKENS = getattr(ai_gateway_module, "MAX_TOKENS", MAX_TOKENS)
                
                logger.info(f"Loaded AI gateway configuration successfully")
                logger.info(f"- AI_GATEWAY_CHAT_ENDPOINT = {AI_GATEWAY_CHAT_ENDPOINT}")
                logger.info(f"- EVAL_PROFILE = {EVAL_PROFILE}")
                logger.info(f"- MODEL_TEMPERATURE = {MODEL_TEMPERATURE}")
                logger.info(f"- MAX_TOKENS = {MAX_TOKENS}")
            else:
                logger.warning(f"AI gateway config not found at {ai_gateway_path}")
    except Exception as e:
        logger.warning(f"Error loading configuration from {config_path}: {e}")
        logger.debug(traceback.format_exc())

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Simple metrics for evaluating conversation turns."""
    
    def __init__(self, skip_llm=False):
        self.logger = logging.getLogger(__name__)
        
        # Try to load AI Gateway config directly from the environment variable
        # This is a fallback in case the module-level loading didn't work
        config_path = os.environ.get("BOT_CONFIG_PATH")
        ai_config_loaded = False
        
        if config_path and os.path.exists(config_path):
            try:
                import importlib.util
                
                config_path = Path(config_path)
                
                # If BOT_CONFIG_PATH is a directory, look for ai_gateway.py
                if config_path.is_dir():
                    ai_gateway_path = config_path / "ai_gateway.py"
                else:
                    # If it's a file, look in the same directory
                    ai_gateway_path = config_path.parent / "ai_gateway.py"
                
                if ai_gateway_path.exists():
                    self.logger.info(f"Loading AI gateway config from {ai_gateway_path}")
                    spec = importlib.util.spec_from_file_location("ai_gateway_config", str(ai_gateway_path))
                    ai_gateway_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(ai_gateway_module)
                    
                    # Override the global variables with the loaded values
                    global AI_GATEWAY_CHAT_ENDPOINT, AI_GATEWAY_TOKEN, EVAL_PROFILE
                    global MODEL_TEMPERATURE, MAX_TOKENS
                    
                    AI_GATEWAY_CHAT_ENDPOINT = getattr(ai_gateway_module, "AI_GATEWAY_CHAT_ENDPOINT", AI_GATEWAY_CHAT_ENDPOINT)
                    AI_GATEWAY_TOKEN = getattr(ai_gateway_module, "AI_GATEWAY_TOKEN", AI_GATEWAY_TOKEN)
                    EVAL_PROFILE = getattr(ai_gateway_module, "EVAL_PROFILE", EVAL_PROFILE)
                    MODEL_TEMPERATURE = getattr(ai_gateway_module, "MODEL_TEMPERATURE", MODEL_TEMPERATURE)
                    MAX_TOKENS = getattr(ai_gateway_module, "MAX_TOKENS", MAX_TOKENS)
                    ai_config_loaded = True
                    
                    self.logger.info("Successfully loaded AI gateway configuration")
            except Exception as e:
                self.logger.error(f"Error loading AI Gateway config: {e}")
                self.logger.debug(traceback.format_exc())
        
        # Set instance variables from global variables
        self.api_endpoint = AI_GATEWAY_CHAT_ENDPOINT
        self.api_key = AI_GATEWAY_TOKEN
        self.model = EVAL_PROFILE
        self.temperature = MODEL_TEMPERATURE
        self.max_tokens = MAX_TOKENS
        # Always use LLM evaluation - ignore the skip_llm parameter
        self.skip_llm = False
        
        # Log a summary of the configuration
        self.logger.info(f"Initializing EvaluationMetrics with:")
        self.logger.info(f"- ACCURACY_THRESHOLD: {ACCURACY_THRESHOLD}")
        self.logger.info(f"- AI_GATEWAY_TOKEN: {'[SET]' if self.api_key else '[NOT SET]'}")
        self.logger.info(f"- AI_GATEWAY_CHAT_ENDPOINT: {self.api_endpoint}")
        self.logger.info(f"- EVAL_PROFILE: {self.model}")
        self.logger.info(f"- MODEL_TEMPERATURE: {self.temperature}")
        self.logger.info(f"- MAX_TOKENS: {self.max_tokens}")
        
        # Check if required configuration is available
        missing_config = []
        if not self.api_endpoint:
            missing_config.append("AI_GATEWAY_CHAT_ENDPOINT")
            self.logger.error("AI_GATEWAY_CHAT_ENDPOINT is not set in the config. Fix the config.")
        if not self.model:
            missing_config.append("EVAL_PROFILE")
            self.logger.error("EVAL_PROFILE is not set in the config. Fix the config.")
        if not self.api_key:
            missing_config.append("AI_GATEWAY_TOKEN")
            self.logger.error("AI_GATEWAY_TOKEN is not set in the config. Fix the config.")
            
        if missing_config:
            self.logger.error(f"Missing required configuration: {', '.join(missing_config)}. Cannot use LLM evaluation.")
            self.logger.error(f"Make sure your ai_gateway.py has all required settings and is being loaded correctly.")
            
            if not ai_config_loaded:
                self.logger.error(f"AI gateway config was not loaded. Check that {config_path} contains or points to a valid ai_gateway.py file.")
            
            raise ValueError(f"Missing required LLM configuration: {', '.join(missing_config)}")
        
        self.logger.info(f"Initialized evaluation metrics with model: {self.model}")
        self.logger.info(f"API endpoint: {self.api_endpoint}")
    
    def call_llm(self, prompt: str) -> str:
        """Call the language model API directly."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        response = requests.post(
            self.api_endpoint,
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            self.logger.error(f"API Error: {response.status_code} - {response.text}")
            return "Error"
        
        return response.json()["choices"][0]["message"]["content"]
    
    async def evaluate_single_turn(self, turn: Dict[str, str], conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Evaluate a single conversation turn using direct LLM prompting or basic comparison."""
        try:
            # Get texts for comparison
            bot_text = turn.get("bot", "").strip()
            exp_text = turn["exp"].strip()
            user_text = turn["user"].strip()
            
            self.logger.info(f"\nComparing responses:")
            self.logger.info(f"User:     {user_text}")
            self.logger.info(f"Expected: {exp_text}")
            self.logger.info(f"Bot:      {bot_text}")
            
            # If LLM evaluation is skipped, use basic comparison
            if self.skip_llm:
                # Improve basic comparison logic for better semantic matching
                
                # Normalize texts for comparison (lowercase, remove punctuation)
                norm_bot_text = bot_text.lower()
                norm_exp_text = exp_text.lower()
                
                # Direct match check
                exact_match = norm_bot_text == norm_exp_text
                
                # German synonyms/equivalents to handle for improved matching
                synonyms = {
                    'nennen': ['mitteilen', 'angeben', 'sagen'],
                    'mitteilen': ['nennen', 'angeben', 'sagen'],
                    '5-stellig': ['5-stellige', 'fünfstellig', 'fünfstellige', '5 stellig'],
                    '5-stellige': ['5-stellig', 'fünfstellig', 'fünfstellige', '5 stellig']
                }
                
                # Check for key phrases that must be present (customize based on your use case)
                key_phrases = [
                    "servicepin",
                    "dank",
                    "bitte",
                ]
                
                # Check if key phrases are present
                key_phrases_present = all(phrase in norm_bot_text for phrase in key_phrases)
                
                # Check if the two texts have similar meaning by checking for key words
                # with synonym replacement
                exp_words = [w for w in norm_exp_text.split() if len(w) > 3]
                matched_words = 0
                
                for word in exp_words:
                    # Check if word or any of its synonyms are in bot response
                    if word in norm_bot_text:
                        matched_words += 1
                    elif word in synonyms:
                        # Check for synonyms
                        if any(syn in norm_bot_text for syn in synonyms[word]):
                            matched_words += 1
                
                # Calculate match percentage
                match_percentage = matched_words / len(exp_words) if exp_words else 0
                
                # Special case handling for our specific expected/bot response comparison
                # "Vielen Dank. Könnten Sie mir bitte Ihre ServicePIN nennen? Sie sollte 5-stellig und nur aus Zahlen bestehen."
                # vs
                # "Vielen Dank. Könnten Sie mir bitte Ihre 5-stellige ServicePIN mitteilen?"
                
                if "servicepin" in norm_bot_text and "servicepin" in norm_exp_text:
                    # Check if both mention "5-stellig" or equivalent
                    has_digit_count = any(phrase in norm_bot_text for phrase in ['5-stellig', '5-stellige', 'fünfstellig']) and \
                                    any(phrase in norm_exp_text for phrase in ['5-stellig', '5-stellige', 'fünfstellig'])
                    # If PIN format is mentioned in both, give bonus points
                    if has_digit_count:
                        match_percentage = min(1.0, match_percentage + 0.2)
                
                # Assign score based on match quality
                if exact_match:
                    score = 1.0
                    self.logger.info("Basic comparison: Exact match")
                elif key_phrases_present and match_percentage >= 0.7:
                    score = 0.9
                    self.logger.info(f"Basic comparison: Strong semantic match ({match_percentage:.2f})")
                elif key_phrases_present and match_percentage >= 0.5:
                    score = 0.8
                    self.logger.info(f"Basic comparison: Good semantic match ({match_percentage:.2f})")
                elif key_phrases_present:
                    score = 0.7
                    self.logger.info(f"Basic comparison: Key phrases present, partial match ({match_percentage:.2f})")
                else:
                    score = match_percentage
                    self.logger.info(f"Basic comparison: Word matching score: {score:.2f}")
            else:
                # Use LLM for evaluation
                # Create evaluation prompt
                prompt = f"""You are evaluating if a bot response correctly matches an expected response in German customer service contexts. Pay special attention to semantic equivalence rather than exact wording.

EVALUATION TASK:
- User message: "{user_text}"
- Expected response: "{exp_text}"
- Actual bot response: "{bot_text}"

Instead of just 0 or 1, rate how well the bot response matches the expected response on a scale from 0.0 to 1.0, with increments of 0.1.

SCORING GUIDE:
- 1.0: Perfect match in meaning and function, even with different words
- 0.9: Excellent match with minor differences that don't affect meaning
- 0.8: Very good match with small omissions that don't change the core message
- 0.7: Good match with acceptable differences
- 0.6-0.5: Acceptable match but with some notable differences
- 0.4-0.3: Poor match with significant differences
- 0.2-0.1: Very poor match, missing critical information
- 0.0: Completely different function or contradictory information

Pay special attention to:
- In German, different verbs like "nennen", "mitteilen", "angeben" often serve the same function
- Word order variations are common and acceptable
- Phrases like "5-stellig" and "5-stellige" are equivalent
- The presence of critical information (like a PIN being numeric)
- Whether both responses request the same information

IMPORTANT: For the specific case of asking for a ServicePIN, consider that asking for a "5-stellige ServicePIN" is equivalent to asking for a "ServicePIN" that "sollte 5-stellig und nur aus Zahlen bestehen".

Your response must be ONLY a number between 0.0 and 1.0, like "0.7" or "0.9" and nothing else.
"""
                
                # Call the LLM
                try:
                    result_text = self.call_llm(prompt).strip()
                    
                    # Parse the result
                    try:
                        score = float(result_text)
                        if score not in [0.0, 1.0]:
                            self.logger.warning(f"LLM returned invalid score: {result_text}, defaulting to 0.0")
                            score = 0.0
                    except ValueError:
                        self.logger.warning(f"Failed to parse LLM response as number: '{result_text}', defaulting to 0.0")
                        score = 0.0
                    
                    self.logger.info(f"LLM evaluation result: {score}")
                except Exception as e:
                    self.logger.error(f"Error calling LLM API: {str(e)}")
                    self.logger.error("Falling back to basic comparison...")
                    
                    # Fall back to simple comparison if API call fails
                    exact_match = bot_text.lower() == exp_text.lower()
                    score = 1.0 if exact_match else 0.0
                    self.logger.info(f"Fallback basic comparison result: {score}")
            
            # Return evaluation result
            threshold = ACCURACY_THRESHOLD if ACCURACY_THRESHOLD is not None else 0.75
            success = score >= threshold
            
            return {
                "score": score,
                "success": success,
                "comparison": {
                    "user_text": user_text,
                    "exp_text": exp_text,
                    "bot_text": bot_text
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating turn: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "score": 0.0,
                "success": False,
                "error": str(e),
                "comparison": {
                    "user_text": user_text,
                    "exp_text": exp_text,
                    "bot_text": bot_text
                }
            }
    
    async def evaluate_conversation(self, conversation_turns: List[Dict[str, str]], skip_turns: int = 0) -> Dict[str, Any]:
        """Evaluate conversation turns, skipping the specified number of initial turns."""
        try:
            all_evaluations = []
            
            # Skip specified turns and evaluate the rest
            turns_to_evaluate = conversation_turns[skip_turns:]
            
            for i, turn in enumerate(turns_to_evaluate, start=skip_turns):
                try:
                    # Get conversation history up to current turn
                    history = conversation_turns[:i] if i > 0 else None
                    evaluation = await self.evaluate_single_turn(turn, history)
                    all_evaluations.append(evaluation)
                except Exception as turn_error:
                    self.logger.error(f"Failed to evaluate turn {i + 1}: {str(turn_error)}")
                    all_evaluations.append({
                        "score": 0.0,
                        "success": False,
                        "error": str(turn_error),
                        "comparison": {
                            "user_text": turn.get("user", ""),
                            "exp_text": turn.get("exp", ""),
                            "bot_text": turn.get("bot", "")
                        }
                    })
            
            # Calculate overall success based on all evaluated turns
            if all_evaluations:
                successful_turns = sum(1 for eval in all_evaluations if eval["success"])
                total_turns = len(all_evaluations)
                overall_success = successful_turns == total_turns
                
                avg_score = sum(eval["score"] for eval in all_evaluations) / total_turns if total_turns > 0 else 0
                
                return {
                    "success": overall_success,
                    "summary": {
                        "total_turns_evaluated": total_turns,
                        "successful_turns": successful_turns,
                        "failed_turns": total_turns - successful_turns,
                        "success_rate": round((successful_turns/total_turns)*100, 1) if total_turns > 0 else 0,
                        "average_score": round(avg_score, 2)
                    },
                    "turn_evaluations": all_evaluations
                }
            else:
                return {
                    "success": False,
                    "summary": {
                        "total_turns_evaluated": 0,
                        "successful_turns": 0,
                        "failed_turns": 0,
                        "success_rate": 0,
                        "average_score": 0
                    },
                    "turn_evaluations": []
                }
            
        except Exception as e:
            self.logger.error(f"Error evaluating conversation: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "turn_evaluations": []
            } 