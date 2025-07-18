import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

import asyncio
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
import traceback
import yaml

from metrics.llm_metrics import EvaluationMetrics

# Default results directory
DEFAULT_RESULTS_DIR = "results"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define results directory if not imported from config
RESULTS_DIR = os.path.join("results")

def load_simulation_results(simulation_file: str) -> List[Dict[str, str]]:
    """Load conversation turns from a simulation results file.
    
    Handles multi-line expected responses and bot responses.
    """
    try:
        turns = []
        
        with open(simulation_file, 'r', encoding='utf-8') as f:
            content = f.read()
            sections = content.split('---')
            
            for section in sections:
                if not section.strip():
                    continue
                
                lines = section.strip().split('\n')
                current_turn = {}
                
                in_exp_block = False
                in_bot_block = False
                for line in lines:
                    if line.startswith('user: '):
                        in_exp_block = False
                        in_bot_block = False
                        current_turn['user'] = line[6:]
                    elif line.startswith('exp: '):
                        in_exp_block = True
                        in_bot_block = False
                        if 'exp' not in current_turn:
                            current_turn['exp'] = []
                        # Add this exp line to the list of expected responses
                        current_turn['exp'].append(line[5:])
                    elif in_exp_block and not line.startswith('bot: '):
                        # Lines following an 'exp:' line are part of the multi-line expected response
                        if 'exp' in current_turn and current_turn['exp']:
                            current_turn['exp'].append(line)
                    elif line.startswith('bot: '):
                        in_exp_block = False
                        in_bot_block = True
                        if 'bot' not in current_turn:
                            current_turn['bot'] = []
                        current_turn['bot'].append(line[5:])
                    elif in_bot_block and not line.startswith('user: ') and not line.startswith('exp: '):
                        # Lines following a 'bot:' line are part of the multi-line bot response
                        if 'bot' in current_turn and current_turn['bot']:
                            # Append to the last bot response
                            current_turn['bot'][-1] = current_turn['bot'][-1] + '\n' + line
                
                # Process multi-line expected response
                if isinstance(current_turn.get('exp'), list):
                    current_turn['exp'] = '\n'.join(current_turn['exp'])
                
                if current_turn:
                    turns.append(current_turn)
            
        logger.debug(f"Loaded {len(turns)} turns from {simulation_file}")
        return turns
        
    except Exception as e:
        logger.error(f"Error loading simulation results from {simulation_file}: {e}")
        logger.error(traceback.format_exc())
        return []

async def evaluate_conversation(conversation_turns: List[Dict[str, str]], skip_turns: int = 0) -> Dict[str, Any]:
    """Evaluate conversation turns using LLM metrics."""
    try:
        # Initialize metrics
        metrics = EvaluationMetrics()
        
        # Process turns to handle multiple bot responses
        processed_turns = []
        for turn in conversation_turns:
            processed_turn = {
                'user': turn.get('user', ''),
                'exp': turn.get('exp', '')
            }
            
            # Handle multiple bot responses by joining them with newlines
            if isinstance(turn.get('bot'), list):
                # Join all bot responses, preserving any existing newlines within each response
                processed_turn['bot'] = '\n'.join(turn.get('bot', []))
            else:
                processed_turn['bot'] = turn.get('bot', '')
            
            processed_turns.append(processed_turn)
            
        # Skip initial turns if specified
        if skip_turns > 0:
            logger.info(f"Skipping first {skip_turns} turns as per configuration")
            processed_turns = processed_turns[skip_turns:]
        
        # Evaluate conversation
        evaluation_results = await metrics.evaluate_conversation(processed_turns, skip_turns)
        
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
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def save_evaluation_results(evaluation_results: Dict[str, Any], simulation_file: str, custom_results_dir: str = None):
    """Save evaluation results to a timestamped folder."""
    try:
        # Create base results directory
        results_dir = Path(custom_results_dir) if custom_results_dir else Path(RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = results_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Generate filename based on simulation file
        sim_path = Path(simulation_file)
        filename = f"evaluation_{sim_path.stem}.txt"
        
        # Format evaluation results
        text_content = []
        text_content.append(f"Evaluation Results for {sim_path.name}")
        text_content.append(f"Timestamp: {timestamp}")
        text_content.append("-" * 50)
        
        # Process summary
        if 'summary' in evaluation_results:
            summary = evaluation_results['summary']
            text_content.append(f"TEST STATUS: {'PASSED' if evaluation_results['success'] else 'FAILED'}")
            text_content.append("-" * 50)
            text_content.append(f"Total turns evaluated: {summary['total_turns_evaluated']}")
            text_content.append(f"Successful turns: {summary['successful_turns']}")
            text_content.append(f"Failed turns: {summary['failed_turns']}")
            text_content.append(f"Success rate: {summary['success_rate']}%")
            text_content.append(f"Average score: {summary['average_score']}")
            text_content.append("-" * 50)
        
        # Process turn evaluations
        if 'turn_evaluations' in evaluation_results:
            text_content.append("\nTurn-by-turn results:")
            for i, eval_result in enumerate(evaluation_results['turn_evaluations'], 1):
                text_content.append(f"\nTurn {i}:")
                if 'comparison' in eval_result:
                    text_content.append(f"User: {eval_result['comparison']['user_text']}")
                    text_content.append(f"Expected: {eval_result['comparison']['exp_text']}")
                    text_content.append(f"Bot: {eval_result['comparison']['bot_text']}")
                text_content.append(f"Score: {eval_result['score']:.2f}")
                text_content.append(f"Success: {'Yes' if eval_result['success'] else 'No'}")
                if 'error' in eval_result:
                    text_content.append(f"Error: {eval_result['error']}")
                text_content.append("-" * 50)
        
        # Save results
        results_file = run_dir / filename
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(text_content))
            
        logger.info(f"Evaluation results saved to {results_file}")
        return str(results_file)
        
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        logger.error(traceback.format_exc())
        return None

def print_evaluation_results(results: Dict[str, Any], output_file: str):
    """Print evaluation results to both console and file."""
    try:
        # Get summary statistics and metadata
        summary = results.get("summary", {})
        total_turns = summary.get("total_turns_evaluated", 0)
        successful_turns = summary.get("successful_turns", 0)
        failed_turns = summary.get("failed_turns", 0)
        success_rate = summary.get("success_rate", 0.0)
        average_score = summary.get("average_score", 0.0)
        
        # Test is failed if any turn fails
        test_passed = failed_turns == 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file = os.path.basename(results.get("input_file", "unknown"))
        
        # Format output
        output = [
            f"Evaluation Results for {input_file}",
            f"Timestamp: {timestamp}",
            "=" * 50,
            f"TEST STATUS: {'PASSED' if test_passed else 'FAILED'}",
            "=" * 50,
            f"Total turns evaluated: {total_turns}",
            f"Successful turns: {successful_turns}",
            f"Failed turns: {failed_turns}",
            f"Success rate: {success_rate:.1f}%",
            f"Average score: {average_score:.2f}",
            "-" * 50,
            "\nTurn-by-turn results:\n"
        ]
        
        # Add turn-by-turn results
        for i, eval_result in enumerate(results.get("turn_evaluations", []), 1):
            comparison = eval_result.get("comparison", {})
            output.extend([
                f"Turn {i}:",
                f"User: {comparison.get('user_text', 'N/A')}",
                f"Expected: {comparison.get('exp_text', 'N/A')}",
                f"Bot: {comparison.get('bot_text', 'N/A')}",
                f"Score: {eval_result.get('score', 0.0):.2f}",
                f"Success: {'Yes' if eval_result.get('success', False) else 'No'}",
                "-" * 50
            ])
        
        # Write to file
        # Ensure output_file is handled as a path
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output))
        
        # Print to console
        print("\n".join(output))
        
    except Exception as e:
        logger.error(f"Error printing evaluation Evaluation result: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def generate_summary(all_evaluations: List[Dict[str, Any]], simulation_files: List[str], custom_results_dir: str = None):
    """Generate summary of test results in the run folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ensure RESULTS_DIR is treated as a Path object
    results_dir = Path(custom_results_dir if custom_results_dir else RESULTS_DIR)
    run_dir = results_dir / f"run_{timestamp}"
    
    # Count successes and failures
    successful_sims = []
    failed_sims = []
    
    for eval_results, sim_file in zip(all_evaluations, simulation_files):
        # A test is successful only if all turns passed
        if 'summary' in eval_results and eval_results['summary']['failed_turns'] == 0:
            successful_sims.append({
                "simulation": os.path.basename(sim_file),
                "evaluation": f"evaluation_{Path(sim_file).stem}.txt",
                "results": eval_results
            })
        else:
            failed_sims.append({
                "simulation": os.path.basename(sim_file),
                "evaluation": f"evaluation_{Path(sim_file).stem}.txt",
                "results": eval_results
            })
    
    # Format summary
    text_content = [
        "Test Summary",
        "=" * 50,
        f"Run timestamp: {timestamp}",
        f"Simulations evaluated: {len(simulation_files)}",
        f"Successful simulations: {len(successful_sims)}",
        f"Failed simulations: {len(failed_sims)}"
    ]
    
    if successful_sims:
        text_content.extend([
            "\nSuccessful simulations:",
            "=" * 50
        ])
        for sim in successful_sims:
            text_content.extend([
                f"Test case: {sim['simulation']}",
                f"Results: {sim['evaluation']}"
            ])
            # Show any turns with low scores but still passed
            if 'turn_evaluations' in sim['results']:
                low_score_turns = [turn for turn in sim['results']['turn_evaluations'] 
                                 if turn.get('score', 1.0) < 0.8]
                if low_score_turns:
                    text_content.append("  Note: Contains turns with scores below 0.8:")
                    for turn in low_score_turns:
                        text_content.append(f"    Score: {turn.get('score', 0):.2f}")
                        if 'comparison' in turn:
                            comp = turn['comparison']
                            text_content.append(f"    Expected: {comp.get('exp_text', 'N/A')}")
                            text_content.append(f"    Bot: {comp.get('bot_text', 'N/A')}")
            text_content.append("-" * 30)
    
    if failed_sims:
        text_content.extend([
            "\nFailed simulations:",
            "=" * 50
        ])
        for sim in failed_sims:
            text_content.extend([
                f"Test case: {sim['simulation']}",
                f"Results: {sim['evaluation']}"
            ])
            if 'summary' in sim['results']:
                summary = sim['results']['summary']
                text_content.extend([
                    f"Success rate: {summary.get('success_rate', 0)}%",
                    f"Average score: {summary.get('average_score', 0)}"
                ])
            
            # Show details of failed turns
            if 'turn_evaluations' in sim['results']:
                failed_turns = [turn for turn in sim['results']['turn_evaluations'] 
                              if not turn.get('success', True)]
                if failed_turns:
                    text_content.append("\nFailed turns:")
                    for i, turn in enumerate(failed_turns, 1):
                        text_content.extend([
                            f"\n  Turn {i}:",
                            f"  Score: {turn.get('score', 0):.2f}"
                        ])
                        if 'comparison' in turn:
                            comp = turn['comparison']
                            text_content.extend([
                                f"  User: {comp.get('user_text', 'N/A')}",
                                f"  Expected: {comp.get('exp_text', 'N/A')}",
                                f"  Bot: {comp.get('bot_text', 'N/A')}"
                            ])
            text_content.append("-" * 30)
    
    # Save summary
    summary_file = run_dir / f"summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(text_content))
    
    # Print summary
    print("\n".join(text_content))
    
    return {
        "total_simulations": len(simulation_files),
        "successful_simulations": len(successful_sims),
        "failed_simulations": len(failed_sims),
        "run_directory": str(run_dir)
    }

async def run_evaluation(simulation_files: List[str], skip_turns: int = 0):
    """Run evaluation on all simulation files."""
    all_evaluations = []
    
    for sim_file in simulation_files:
        logger.info(f"\nEvaluating simulation file: {sim_file}")
        
        # Load simulation results
        conversation_turns = load_simulation_results(sim_file)
        if not conversation_turns:
            logger.error(f"No conversation turns found in {sim_file}")
            continue
        
        # Evaluate conversation
        evaluation_results = await evaluate_conversation(conversation_turns, skip_turns)
        
        # Add metadata
        evaluation_results["input_file"] = sim_file
        evaluation_results["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        all_evaluations.append(evaluation_results)
        
        # Generate results filename
        timestamp = evaluation_results.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        results_dir = Path(RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        results_filename = results_dir / f"evaluation_{os.path.basename(sim_file)}_{timestamp}.txt"
        
        # Save and print results
        print_evaluation_results(evaluation_results, results_filename)
    
    # Generate and print overall summary
    if all_evaluations:
        summary_info = generate_summary(all_evaluations, simulation_files)
        print("\nOverall Summary:")
        print(f"Total simulations: {summary_info['total_simulations']}")
        if 'overall_results' in summary_info:
            results = summary_info['overall_results']
            print(f"Total turns: {results['total_turns']}")
            print(f"Successful turns: {results['successful_turns']}")
            print(f"Overall success rate: {results['success_rate']}%")
            print(f"Overall average score: {results['average_score']}")

def format_evaluation_results(results: Dict[str, Any]) -> str:
    """Format evaluation results as a string."""
    lines = []
    
    # Add header
    sim_file = os.path.basename(results.get('input_file', 'unknown'))
    timestamp = results.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
    lines.append(f"Evaluation Results for {sim_file}")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("=" * 50)
    
    # Add test status
    passed = results.get('success', False)
    lines.append(f"TEST STATUS: {'PASSED' if passed else 'FAILED'}")
    lines.append("=" * 50)
    
    # Add summary statistics
    if 'summary' in results:
        summary = results['summary']
        lines.append(f"Total turns evaluated: {summary.get('total_turns_evaluated', 0)}")
        lines.append(f"Successful turns: {summary.get('successful_turns', 0)}")
        lines.append(f"Failed turns: {summary.get('failed_turns', 0)}")
        lines.append(f"Success rate: {summary.get('success_rate', 0)}%")
        lines.append(f"Average score: {summary.get('average_score', 0)}")
        lines.append("-" * 50)
    
    # Add turn-by-turn results
    lines.append("\nTurn-by-turn results:\n")
    
    for i, eval_result in enumerate(results.get('turn_evaluations', []), 1):
        comparison = eval_result.get('comparison', {})
        lines.append(f"Turn {i}:")
        lines.append(f"User: {comparison.get('user_text', 'N/A')}")
        lines.append(f"Expected: {comparison.get('exp_text', 'N/A')}")
        lines.append(f"Bot: {comparison.get('bot_text', 'N/A')}")
        lines.append(f"Score: {eval_result.get('score', 0.0):.2f}")
        lines.append(f"Success: {'Yes' if eval_result.get('success', False) else 'No'}")
        if 'error' in eval_result:
            lines.append(f"Error: {eval_result['error']}")
        lines.append("-" * 50)
    
    return "\n".join(lines)

def format_summary(all_results: List[Dict[str, Any]], simulation_files: List[str], timestamp: str) -> str:
    """Format summary as a string."""
    lines = []
    
    # Default values if config modules not available
    AI_GATEWAY_CHAT_ENDPOINT = "N/A"
    EVAL_PROFILE = "N/A"
    MODEL_TEMPERATURE = "N/A"
    MAX_TOKENS = "N/A"
    ACCURACY_THRESHOLD = "N/A"
    
    # Try to get values from environment
    config_path = os.environ.get("BOT_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        try:
            import importlib.util
            
            config_path = Path(config_path)
            
            # If BOT_CONFIG_PATH is a directory, look for base.py and ai_gateway.py
            if config_path.is_dir():
                # Load base.py for accuracy threshold
                base_path = config_path / "base.py"
                if base_path.exists():
                    spec = importlib.util.spec_from_file_location("base_config", str(base_path))
                    base_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(base_module)
                    ACCURACY_THRESHOLD = getattr(base_module, "ACCURACY_THRESHOLD", ACCURACY_THRESHOLD)
                
                # Load ai_gateway.py for gateway settings
                ai_gateway_path = config_path / "ai_gateway.py"
                if ai_gateway_path.exists():
                    spec = importlib.util.spec_from_file_location("ai_gateway_config", str(ai_gateway_path))
                    ai_gateway_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(ai_gateway_module)
                    AI_GATEWAY_CHAT_ENDPOINT = getattr(ai_gateway_module, "AI_GATEWAY_CHAT_ENDPOINT", AI_GATEWAY_CHAT_ENDPOINT)
                    EVAL_PROFILE = getattr(ai_gateway_module, "EVAL_PROFILE", EVAL_PROFILE)
                    MODEL_TEMPERATURE = getattr(ai_gateway_module, "MODEL_TEMPERATURE", MODEL_TEMPERATURE)
                    MAX_TOKENS = getattr(ai_gateway_module, "MAX_TOKENS", MAX_TOKENS)
            # If BOT_CONFIG_PATH is a file, use its directory to find ai_gateway.py
            else:
                config_dir = config_path.parent
                
                # Try to load ai_gateway.py from the same directory
                ai_gateway_path = config_dir / "ai_gateway.py"
                if ai_gateway_path.exists():
                    spec = importlib.util.spec_from_file_location("ai_gateway_config", str(ai_gateway_path))
                    ai_gateway_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(ai_gateway_module)
                    AI_GATEWAY_CHAT_ENDPOINT = getattr(ai_gateway_module, "AI_GATEWAY_CHAT_ENDPOINT", AI_GATEWAY_CHAT_ENDPOINT)
                    EVAL_PROFILE = getattr(ai_gateway_module, "EVAL_PROFILE", EVAL_PROFILE)
                    MODEL_TEMPERATURE = getattr(ai_gateway_module, "MODEL_TEMPERATURE", MODEL_TEMPERATURE)
                    MAX_TOKENS = getattr(ai_gateway_module, "MAX_TOKENS", MAX_TOKENS)
                
                # Try to load accuracy threshold from the config file itself
                try:
                    spec = importlib.util.spec_from_file_location("config_file", str(config_path))
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    ACCURACY_THRESHOLD = getattr(config_module, "ACCURACY_THRESHOLD", ACCURACY_THRESHOLD)
                except:
                    # Failed to load from config file, no problem
                    pass
        except Exception as e:
            logger.warning(f"Error loading config values for summary: {e}")
            logger.debug(traceback.format_exc())
        try:
            import importlib.util
            
            # Handle directory vs file path
            config_path_obj = Path(config_path)
            config_dir = config_path_obj
            
            if config_path_obj.is_dir():
                # If it's a directory, use base.py
                config_path_obj = config_path_obj / "base.py"
                config_dir = config_path_obj.parent
                logger.info(f"Looking for base.py in directory: {config_path_obj}")
            else:
                # If it's a file, get the parent directory
                config_dir = config_path_obj.parent
            
            # Load base config
            if config_path_obj.exists():
                logger.info(f"Loading base config from: {config_path_obj}")
                spec = importlib.util.spec_from_file_location("bot_config", str(config_path_obj))
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                
                # Get accuracy threshold
                ACCURACY_THRESHOLD = getattr(config_module, "ACCURACY_THRESHOLD", ACCURACY_THRESHOLD)
                logger.info(f"Loaded ACCURACY_THRESHOLD: {ACCURACY_THRESHOLD}")
            
            # Always try to load AI gateway config
            ai_gateway_path = config_dir / "ai_gateway.py"
            if ai_gateway_path.exists():
                logger.info(f"Loading AI gateway config from: {ai_gateway_path}")
                spec = importlib.util.spec_from_file_location("ai_gateway", str(ai_gateway_path))
                ai_gateway_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ai_gateway_module)
                
                # Get AI gateway values
                AI_GATEWAY_CHAT_ENDPOINT = getattr(ai_gateway_module, "AI_GATEWAY_CHAT_ENDPOINT", AI_GATEWAY_CHAT_ENDPOINT)
                EVAL_PROFILE = getattr(ai_gateway_module, "EVAL_PROFILE", EVAL_PROFILE)
                MODEL_TEMPERATURE = getattr(ai_gateway_module, "MODEL_TEMPERATURE", MODEL_TEMPERATURE)
                MAX_TOKENS = getattr(ai_gateway_module, "MAX_TOKENS", MAX_TOKENS)
                
                logger.info(f"Successfully loaded AI gateway configuration:")
                logger.info(f"- AI_GATEWAY_CHAT_ENDPOINT: {AI_GATEWAY_CHAT_ENDPOINT}")
                logger.info(f"- EVAL_PROFILE: {EVAL_PROFILE}")
                logger.info(f"- MODEL_TEMPERATURE: {MODEL_TEMPERATURE}")
                logger.info(f"- MAX_TOKENS: {MAX_TOKENS}")
            else:
                logger.error(f"AI gateway config not found at {ai_gateway_path}")
        except Exception as e:
            logger.error(f"Could not load configuration values for summary: {e}")
            logger.debug(traceback.format_exc())
    
    # Get metrics type from first result (all should be the same in a run)
    metrics_type = all_results[0].get('metrics_type', 'EvaluationMetrics') if all_results else 'EvaluationMetrics'
    
    lines.append("Test Configuration")
    lines.append("=" * 50)
    lines.append(f"Metrics Type: {metrics_type}")
    lines.append("")
    lines.append("AI Gateway Configuration:")
    lines.append(f"- Endpoint: {AI_GATEWAY_CHAT_ENDPOINT}")
    lines.append(f"- Evaluation Profile: {EVAL_PROFILE}")
    lines.append(f"- Model Temperature: {MODEL_TEMPERATURE}")
    lines.append(f"- Max Tokens: {MAX_TOKENS}")
    lines.append(f"- Accuracy Threshold: {ACCURACY_THRESHOLD}")
    lines.append("-" * 50)
    lines.append("")
    
    # Count evaluation errors
    eval_errors = sum(1 for r in all_results if 'error' in r)
    
    # Total counts
    total_sims = len(simulation_files)
    successful_sims = sum(1 for r in all_results if r.get('success', False))
    failed_sims = total_sims - successful_sims
    
    lines.append("Test Summary")
    lines.append("=" * 50)
    lines.append(f"Run timestamp: {timestamp}")
    lines.append(f"Simulations evaluated: {total_sims}")
    lines.append(f"Successful simulations: {successful_sims}")
    lines.append(f"Failed simulations: {failed_sims}")
    if eval_errors > 0:
        lines.append(f"Evaluation errors: {eval_errors}")
    lines.append("")
    
    # List evaluation errors first if any
    if eval_errors > 0:
        lines.append("\nEvaluation Errors:")
        lines.append("=" * 50)
        for result, sim_file in zip(all_results, simulation_files):
            if 'error' in result:
                lines.append(f"Test case: {os.path.basename(sim_file)}")
                error_msg = result['error']
                if "API Error:" in error_msg:
                    # Extract and format API error details
                    error_parts = error_msg.split("API Error:", 1)
                    if len(error_parts) > 1:
                        status_code = error_parts[1].split(" - ")[0].strip()
                        error_details = error_parts[1].split(" - ", 1)[1].strip() if " - " in error_parts[1] else ""
                        lines.append(f"API Error:")
                        lines.append(f"  Status Code: {status_code}")
                        lines.append(f"  Error Details: {error_details}")
                    else:
                        lines.append(f"Error: {error_msg}")
                else:
                    lines.append(f"Error: {error_msg}")
                lines.append("-" * 30)
    
    # List successful simulations with details
    if successful_sims > 0:
        lines.append("\nSuccessful simulations:")
        lines.append("=" * 50)
        for result, sim_file in zip(all_results, simulation_files):
            if result.get('success', False):
                lines.append(f"Test case: {os.path.basename(sim_file)}")
                if 'turn_evaluations' in result:
                    # Show any turns that had low scores but still passed
                    low_score_turns = [turn for turn in result['turn_evaluations'] 
                                     if turn.get('score', 1.0) < 0.8]
                    if low_score_turns:
                        lines.append("  Note: Contains turns with scores below 0.8:")
                        for turn in low_score_turns:
                            lines.append(f"    Score: {turn.get('score', 0):.2f}")
                            if 'comparison' in turn:
                                comp = turn['comparison']
                                lines.append(f"    Expected: {comp.get('exp_text', 'N/A')}")
                                lines.append(f"    Bot: {comp.get('bot_text', 'N/A')}")
                lines.append("-" * 30)
    
    # List failed simulations with detailed failure information
    failed_without_errors = sum(1 for r in all_results if not r.get('success', False) and 'error' not in r)
    if failed_without_errors > 0:
        lines.append("\nFailed simulations:")
        lines.append("=" * 50)
        for result, sim_file in zip(all_results, simulation_files):
            if not result.get('success', False) and 'error' not in result:
                lines.append(f"Test case: {os.path.basename(sim_file)}")
                if 'summary' in result:
                    summary = result['summary']
                    lines.append(f"Success rate: {summary.get('success_rate', 0)}%")
                    lines.append(f"Average score: {summary.get('average_score', 0)}")
                
                # Show details of failed turns
                if 'turn_evaluations' in result:
                    failed_turns = [turn for turn in result['turn_evaluations'] 
                                  if not turn.get('success', True)]
                    if failed_turns:
                        lines.append("\nFailed turns:")
                        for i, turn in enumerate(failed_turns, 1):
                            lines.append(f"\n  Turn {i}:")
                            lines.append(f"  Score: {turn.get('score', 0):.2f}")
                            if 'comparison' in turn:
                                comp = turn['comparison']
                                lines.append(f"  User: {comp.get('user_text', 'N/A')}")
                                lines.append(f"  Expected: {comp.get('exp_text', 'N/A')}")
                                lines.append(f"  Bot: {comp.get('bot_text', 'N/A')}")
                lines.append("-" * 30)
    
    return "\n".join(lines)

def load_config(config_path: str) -> dict:
    """Load configuration from Python or YAML file.
    
    Args:
        config_path: Path to the configuration file or directory
        
    Returns:
        dict: Loaded configuration
    """
    global RESULTS_DIR
    
    try:
        # Handle directory (use base.py by default if it's a directory)
        config_path = Path(config_path).resolve()  # Convert to absolute path
        config_dir = config_path
        
        if config_path.is_dir():
            config_dir = config_path  # Remember the directory for resolving relative paths
            config_path = config_path / "base.py"
            logger.info(f"Using base.py from directory: {config_path}")
        else:
            config_dir = config_path.parent  # Remember parent dir for resolving relative paths
        
        # DO NOT set BOT_CONFIG_PATH here - it's set by the caller before this function is called
        # This function should only load the configuration
        
        # Python file loading
        if config_path.suffix == ".py":
            # Import Python file as module
            import importlib.util
            spec = importlib.util.spec_from_file_location("config_module", str(config_path))
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Extract configuration as dictionary
            config = {}
            for key in dir(config_module):
                if not key.startswith("__"):
                    config[key] = getattr(config_module, key)
                    
            # Look for additional files in the same directory
            try:
                ai_gateway_path = config_dir / "ai_gateway.py"
                if ai_gateway_path.exists():
                    logger.info(f"Loading AI gateway config from {ai_gateway_path}")
                    spec = importlib.util.spec_from_file_location("ai_gateway_module", str(ai_gateway_path))
                    ai_gateway_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(ai_gateway_module)
                    
                    # Add AI gateway config to our local config dictionary
                    config["ai_gateway"] = {}
                    for key in dir(ai_gateway_module):
                        if not key.startswith("__"):
                            config["ai_gateway"][key] = getattr(ai_gateway_module, key)
                            
                    # Also add these directly to the config object for backward compatibility
                    for key in dir(ai_gateway_module):
                        if not key.startswith("__"):
                            if key not in config:  # Don't override existing keys
                                config[key] = getattr(ai_gateway_module, key)
                    
                    # Log the loaded configuration
                    logger.info("Successfully loaded AI gateway configuration:")
                    logger.info(f"- AI_GATEWAY_CHAT_ENDPOINT: {config.get('AI_GATEWAY_CHAT_ENDPOINT')}")
                    logger.info(f"- EVAL_PROFILE: {config.get('EVAL_PROFILE')}")
                    logger.info(f"- MODEL_TEMPERATURE: {config.get('MODEL_TEMPERATURE')}")
                    logger.info(f"- MAX_TOKENS: {config.get('MAX_TOKENS')}")
                else:
                    logger.error(f"AI gateway config not found at {ai_gateway_path}")
                    logger.error("AI gateway config is required for LLM evaluation. Cannot continue without it.")
                    raise FileNotFoundError(f"AI gateway config file not found: {ai_gateway_path}")
            except Exception as e:
                logger.error(f"Could not load AI gateway config: {e}")
                logger.error(traceback.format_exc())
                raise
                
            # Resolve relative paths in config
            if "RESULTS_DIR" in config:
                results_dir_value = config["RESULTS_DIR"]
                if isinstance(results_dir_value, str) and results_dir_value.startswith("./"):
                    # Convert relative path to absolute using config_dir
                    results_dir_path = config_dir / results_dir_value[2:]  # Remove './'
                    config["RESULTS_DIR"] = str(results_dir_path)
                    RESULTS_DIR = str(results_dir_path)
                    logger.info(f"Resolved relative RESULTS_DIR to: {RESULTS_DIR}")
                else:
                    RESULTS_DIR = config["RESULTS_DIR"]
                    logger.info(f"Using RESULTS_DIR from config: {RESULTS_DIR}")
            
            # Resolve relative SIMULATIONS_DIR
            if "SIMULATIONS_DIR" in config:
                sim_dir_value = config["SIMULATIONS_DIR"]
                if isinstance(sim_dir_value, str) and sim_dir_value.startswith("./"):
                    # Convert relative path to absolute using config_dir
                    sim_dir_path = config_dir / sim_dir_value[2:]  # Remove './'
                    config["SIMULATIONS_DIR"] = str(sim_dir_path)
                    logger.info(f"Resolved relative SIMULATIONS_DIR to: {config['SIMULATIONS_DIR']}")
            
            return config
        # YAML file loading
        elif config_path.suffix in [".yml", ".yaml"]:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
                # Set results dir if available
                if "RESULTS_DIR" in config:
                    RESULTS_DIR = config["RESULTS_DIR"]
                    logger.info(f"Using RESULTS_DIR from config: {RESULTS_DIR}")
                    
                return config
        else:
            logger.warning(f"Unsupported config file type: {config_path.suffix}. Expected .py, .yml, or .yaml")
            return {}
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        logger.error(traceback.format_exc())
        return {}

def main():
    parser = argparse.ArgumentParser(description='Evaluate simulated conversations')
    parser.add_argument('--config', type=str, required=True, 
                      help='Path to configuration file or directory (if directory, uses base.py)')
    parser.add_argument('--simulations', nargs='+', required=False, 
                      help='Paths to simulation result files to evaluate (overrides config if provided)')
    parser.add_argument('--simulations-dir', type=str, required=False, 
                      help='Directory containing simulation result files (alternative or addition to --simulations)')
    parser.add_argument('--results-dir', type=str, required=False, 
                      help='Directory to save evaluation results (overrides config RESULTS_DIR). Will be created if it does not exist.')
    parser.add_argument('--skip-turns', type=int, default=None, 
                      help='Number of initial turns to skip in evaluation (overrides config if provided)')
    # LLM evaluation is always used, so this parameter is no longer needed
    # parser.add_argument('--skip-llm', action='store_true', 
    #                   help='Skip LLM evaluation and only do basic comparison (useful if LLM API is unavailable)')
    parser.add_argument('--postfix', type=str, default='', 
                      help='Postfix to add to the result filenames')
    parser.add_argument('--verbose', '-v', action='count', default=0, 
                      help='Increase verbosity level (-v for INFO, -vv for DEBUG)')
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
    
    try:
        # Process config path
        config_path = Path(args.config).resolve()
        if config_path.is_dir():
            # If it's a directory, we'll use the base.py file but set BOT_CONFIG_PATH to the directory
            logger.info(f"Config is a directory, using {config_path}/base.py but setting BOT_CONFIG_PATH to the directory")
            # Set BOT_CONFIG_PATH to the directory for metrics to find ai_gateway.py
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
        
        # Use command line arguments if provided, otherwise use config
        simulations = args.simulations or []
        
        # Handle simulations directory if provided
        if args.simulations_dir:
            simulations_dir = Path(args.simulations_dir)
            logger.info(f"Using simulations directory from command line: {simulations_dir}")
        else:
            # Use simulations directory from config if not provided in args
            simulations_dir_from_config = config.get('SIMULATIONS_DIR', None)
            
            if simulations_dir_from_config:
                simulations_dir = Path(simulations_dir_from_config)
                logger.info(f"Using simulations directory from config: {simulations_dir}")
            else:
                # Try to find the simulated_conversations directory relative to config directory
                config_path = Path(args.config)
                if config_path.is_file():
                    config_dir = config_path.parent
                else:
                    config_dir = config_path
                
                # Check for default locations
                possible_sim_dirs = [
                    config_dir / "simulated_conversations",
                    config_dir.parent / "simulated_conversations"
                ]
                
                for possible_dir in possible_sim_dirs:
                    if possible_dir.exists() and possible_dir.is_dir():
                        simulations_dir = possible_dir
                        logger.info(f"Found simulations directory: {simulations_dir}")
                        break
                else:
                    simulations_dir = None
        
        # Process simulations directory if we have one
        if simulations_dir and simulations_dir.exists() and simulations_dir.is_dir():
            # Get all simulation files in the directory
            sim_files = list(simulations_dir.glob("simulation_*.txt"))
            if sim_files:
                # Add files while eliminating duplicates
                sim_paths = [str(f) for f in sim_files]
                for path in sim_paths:
                    if path not in simulations:
                        simulations.append(path)
                logger.info(f"Found {len(sim_files)} simulation files in {simulations_dir}")
            else:
                logger.warning(f"No simulation files found in {simulations_dir}")
        elif simulations_dir:
            logger.warning(f"Simulations directory not found: {simulations_dir}")
            # Create the directory if it doesn't exist
            try:
                simulations_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created simulations directory: {simulations_dir}")
            except Exception as e:
                logger.warning(f"Could not create simulations directory: {e}")
                logger.debug(traceback.format_exc())
        
        # Fall back to config if no simulations found yet
        if not simulations:
            simulations = config.get('evaluation', {}).get('simulation_files', [])
        
        skip_turns = args.skip_turns if args.skip_turns is not None else config.get('evaluation', {}).get('skip_turns', 0)
        
        # Handle results directory
        global RESULTS_DIR
        results_dir = args.results_dir if args.results_dir else config.get('RESULTS_DIR', DEFAULT_RESULTS_DIR)
        RESULTS_DIR = results_dir  # Update global variable for other functions to use
        
        if not simulations:
            logger.error("No simulation files provided. Specify with --simulations, --simulations-dir, or in config file.")
            sys.exit(1)
            
        # Create timestamped run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure results_dir is a Path object
        results_dir_path = Path(results_dir)
        run_dir = results_dir_path / f"run_{timestamp}_{args.postfix}" if args.postfix else results_dir_path / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting evaluation with config: {args.config}")
        logger.info(f"Results will be saved to: {run_dir.absolute()}")
        logger.info(f"Skipping first {skip_turns} turns in each conversation")
        logger.info(f"Will process {len(simulations)} simulation files")
        
        all_results = []
        
        for simulation_file in simulations:
            # Load simulation results
            logger.info(f"Processing simulation file: {simulation_file}")
            turns = load_simulation_results(simulation_file)
            if not turns:
                logger.error(f"Failed to load simulation results from {simulation_file}")
                continue
                
            # Evaluate conversation
            logger.info(f"Evaluating {len(turns)} turns from {simulation_file}")
            logger.info("Using LLM evaluation for accurate semantic matching")
            results = asyncio.run(evaluate_conversation(turns, skip_turns))
            if not results:
                logger.error(f"Failed to evaluate conversation from {simulation_file}")
                continue
                
            # Add metadata
            results["input_file"] = simulation_file
            results["timestamp"] = timestamp
            
            # Generate results filename in the run folder
            sim_name = Path(simulation_file).stem
            results_filename = run_dir / f"evaluation_{sim_name}_{args.postfix}.txt" if args.postfix else run_dir / f"evaluation_{sim_name}.txt"
            
            # Save evaluation results
            with open(results_filename, 'w', encoding='utf-8') as f:
                f.write(format_evaluation_results(results))
            
            logger.info(f"Saved evaluation results to {results_filename}")
            all_results.append(results)
        
        # Generate and save summary
        if all_results:
            summary_filename = run_dir / (f"summary_{args.postfix}.txt" if args.postfix else "summary.txt")
            with open(summary_filename, 'w', encoding='utf-8') as f:
                f.write(format_summary(all_results, simulations, timestamp))
            
            logger.info(f"Saved evaluation summary to {summary_filename}")
            print(f"\nEvaluation complete. Results saved in: {run_dir}")
            
            # Print overall success status
            successful = sum(1 for result in all_results if result.get('success', False))
            total = len(all_results)
            print(f"Overall status: {successful}/{total} simulations passed")
            
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()