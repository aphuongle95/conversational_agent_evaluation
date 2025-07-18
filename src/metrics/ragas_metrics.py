import os
import json
from typing import List, Dict, Any, Optional
from datasets import Dataset
from config.base import ACCURACY_THRESHOLD
from config.ai_gateway import (
    AI_GATEWAY_TOKEN,
    AI_GATEWAY_ENDPOINT,
    EVAL_PROFILE,
    MODEL_TEMPERATURE,
    MAX_TOKENS
)
from ragas.metrics import AspectCritic
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from dataclasses import dataclass
from conversation.conversation_handler import ConversationScript
from utils.helpers import replace_special_chars
import logging
import sys
import asyncio
import requests
import traceback
import nest_asyncio
from httpx import ConnectError, ReadTimeout
from openai import APIError, RateLimitError

# Disable Ragas telemetry
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

# Enable nested event loops
nest_asyncio.apply()

# Set up logging to output to stdout with maximum verbosity
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Configure module loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add detailed request logging for OpenAI
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.DEBUG)

# Add httpx logging for detailed HTTP request/response logging
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.DEBUG)

# Add Langchain logging
langchain_logger = logging.getLogger("langchain")
langchain_logger.setLevel(logging.DEBUG)

# Add Ragas logging
ragas_logger = logging.getLogger("ragas")
ragas_logger.setLevel(logging.DEBUG)

@dataclass
class RagasMetrics:
    """Wrapper for Ragas AspectCritic metric for multi-turn conversation evaluation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        try:
            # Initialize OpenAI client
            self.logger.info("\nInitializing chat model with configuration:")
            self.logger.info(f"API Endpoint: {AI_GATEWAY_ENDPOINT}")
            self.logger.info(f"Model: {EVAL_PROFILE}")
            self.logger.info(f"Temperature: {MODEL_TEMPERATURE}")
            
            self.chat_model = ChatOpenAI(
                api_key=AI_GATEWAY_TOKEN,
                base_url=AI_GATEWAY_ENDPOINT,
                model=EVAL_PROFILE,
                temperature=MODEL_TEMPERATURE,
                verbose=True
            )
            
            # Log the model configuration
            self.logger.info(f"Chat model configuration: {self.chat_model}")
            
            # Initialize LangchainLLMWrapper
            self.llm_wrapper = LangchainLLMWrapper(langchain_llm=self.chat_model)
            self.logger.info(f"LLM wrapper configuration: {self.llm_wrapper}")
            
            # Initialize AspectCritic for scoring
            self.scorer = AspectCritic(
                name="answer_accuracy",
                definition="Is the latest bot answer reasonably aligned with the information in the reference answer?",
                llm=self.llm_wrapper
            )
            
            self.logger.info(f"Scorer configuration: {self.scorer}")
            
        except Exception as e:
            self.logger.error(f"Error initializing RagasMetrics: {str(e)}")
            raise
    
    async def evaluate_single_turn(self, turn: Dict[str, str], conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Evaluate a single conversation turn with its history."""
        try:
            # Build conversation history messages
            messages = []
            if conversation_history:
                self.logger.info("\nConversation history:")
                for hist_turn in conversation_history:
                    self.logger.info(f"User: {hist_turn['user']}")
                    self.logger.info(f"Bot: {hist_turn['exp']}")
                    messages.extend([
                        HumanMessage(content=hist_turn["user"]),
                        AIMessage(content=hist_turn["exp"])
                    ])
            
            # Add current turn
            self.logger.info("\nCurrent turn:")
            self.logger.info(f"User: {turn['user']}")
            self.logger.info(f"Expected: {turn['exp']}")
            self.logger.info(f"Bot: {turn.get('bot', '')}")
            
            messages.extend([
                HumanMessage(content=turn["user"]),
                AIMessage(content=turn["exp"])
            ])
            
            # Log exact comparison for current turn
            bot_text = turn.get("bot", "")  # Use actual bot response if available
            exp_text = turn["exp"]
            self.logger.info(f"\nComparing responses:")
            self.logger.info(f"Bot:      {replace_special_chars(bot_text)}")
            self.logger.info(f"Expected: {replace_special_chars(exp_text)}")
            
            sample = MultiTurnSample(
                user_input=messages,
                reference=turn["exp"]
            )
            
            # Log the sample being evaluated
            self.logger.info("\nEvaluating sample:")
            self.logger.info(f"Messages: {[msg.content for msg in messages]}")
            self.logger.info(f"Reference: {sample.reference}")
            
            # Set up event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Score the turn
            self.logger.info("\nCalling AspectCritic scorer...")
            self.logger.info(f"Scorer definition: {self.scorer.definition}")
            result = await self.scorer.multi_turn_ascore(sample)
            self.logger.info(f"Scorer result: {result}")
            self.logger.info(f"Success threshold: {ACCURACY_THRESHOLD}")
            self.logger.info(f"Success: {result >= ACCURACY_THRESHOLD}")
            
            return {
                "score": result,
                "success": result >= ACCURACY_THRESHOLD,
                "comparison": {
                    "user_text": turn["user"],
                    "exp_text": turn["exp"],
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
                    "user_text": turn["user"],
                    "exp_text": turn["exp"],
                    "bot_text": turn.get("bot", "")
                }
            }
    
    async def evaluate_conversation(self, conversation_turns: List[Dict[str, str]], skip_turns: int = 0) -> Dict[str, Any]:
        """Evaluate conversation turns, skipping the specified number of initial turns."""
        try:
            all_evaluations = []
            overall_success = True
            
            # Skip specified turns and evaluate the rest
            turns_to_evaluate = conversation_turns[skip_turns:]
            
            for i, turn in enumerate(turns_to_evaluate, start=skip_turns):
                try:
                    # Get conversation history up to current turn
                    history = conversation_turns[:i] if i > 0 else None
                    evaluation = await self.evaluate_single_turn(turn, history)
                    all_evaluations.append(evaluation)
                    # Don't update overall_success here - we want to evaluate all turns
                except Exception as turn_error:
                    self.logger.error(f"Failed to evaluate turn {i + 1}: {str(turn_error)}")
                    all_evaluations.append({
                        "score": 0.0,
                        "success": False,
                        "turn_data": turn,
                        "error": str(turn_error)
                    })
                    # Don't update overall_success here - we want to evaluate all turns
            
            # Calculate overall success based on all evaluated turns
            if all_evaluations:
                successful_turns = sum(1 for eval in all_evaluations if eval["success"])
                overall_success = successful_turns == len(all_evaluations)
            
            return {
                "success": overall_success,
                "total_turns": len(conversation_turns),
                "skipped_turns": skip_turns,
                "evaluated_turns": len(all_evaluations),
                "turn_evaluations": all_evaluations
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating conversation: {str(e)}")
            return {
                "success": False,
                "total_turns": len(conversation_turns),
                "skipped_turns": skip_turns,
                "evaluated_turns": 0,
                "turn_evaluations": [],
                "error": str(e)
            } 