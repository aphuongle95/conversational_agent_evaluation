import asyncio
import json
import logging
import argparse
import sys
import glob
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

import importlib.util
from api.bot_client_botario_gpt import BotClient

import traceback
import os

logger = logging.getLogger(__name__)

def read_conversation_script(file_path: str) -> List[Dict[str, str]]:
    """Read a conversation script file and return a list of turns.
    
    Args:
        file_path: Path to the conversation script file
        
    Returns:
        List of turns, where each turn is a dict with 'user' and 'exp' keys
    """
    turns = []
    current_turn = None
    current_exp = []
    in_exp_block = False  # Track if we're in an exp block
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    # Empty line within exp block should be preserved
                    if in_exp_block and current_turn:
                        current_exp.append('')
                    continue
                
                if line == '---':
                    in_exp_block = False
                    continue
                    
                if line.startswith('user: '):
                    in_exp_block = False
                    if current_turn:
                        # Join multiline exp with newlines
                        current_turn['exp'] = '\n'.join(current_exp) if current_exp else ''
                        turns.append(current_turn)
                    current_turn = {'user': line[6:], 'exp': ''}
                    current_exp = []
                elif line.startswith('exp: '):
                    # Remove 'exp: ' prefix and add to current exp
                    current_exp.append(line[5:])
                    in_exp_block = True
                elif in_exp_block and current_turn:
                    # Any line that follows an "exp:" line and isn't another command is part of the multi-line exp
                    current_exp.append(line)
                    
            if current_turn:
                # Join final multiline exp
                current_turn['exp'] = '\n'.join(current_exp) if current_exp else ''
                turns.append(current_turn)
                
        return turns
    except Exception as e:
        logger.error(f"Error reading conversation script: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def create_simulation_file(test_file: str, run_number: int, postfix: str = '', custom_results_dir: str = None, config_module=None):
    """Create a simulation results file and return the file path and handle.
    
    Args:
        test_file: Path to the test file
        run_number: The run number for this simulation
        postfix: Optional postfix to add to the filename
        custom_results_dir: Optional custom directory to save results
        config_module: Optional config module with SIMULATIONS_DIR
    
    Returns:
        Tuple of (file_path, file_handle) or (None, None) if creation failed
    """
    try:
        # Determine results directory
        if custom_results_dir:
            # Use explicit results directory provided by command line
            results_dir = Path(custom_results_dir)
            logger.info(f"Using custom results directory: {results_dir}")
        elif config_module and hasattr(config_module, 'SIMULATIONS_DIR'):
            # Use SIMULATIONS_DIR from config - already resolved to absolute path in run_single_simulation
            results_dir = Path(config_module.SIMULATIONS_DIR)
            logger.info(f"Using SIMULATIONS_DIR from config: {results_dir}")
        else:
            # Get the project directory from the test file path
            project_dir = Path(test_file).parent.parent
            results_dir = project_dir / "simulated_conversations"
            logger.info(f"Using default simulated_conversations directory: {results_dir}")
            
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        test_name = Path(test_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_{timestamp}_{test_name}_run{run_number}{postfix}.txt"
        
        # Create file
        results_file = results_dir / filename
        file_handle = open(results_file, 'w', encoding='utf-8')
        
        logger.info(f"Created simulation file at {results_file}")
        return str(results_file), file_handle
        
    except Exception as e:
        logger.error(f"Error creating simulation file: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def write_turn_to_file(file_handle, turn: Dict[str, Any]):
    """Write a single conversation turn to the file.
    
    Args:
        file_handle: Open file handle to write to
        turn: The conversation turn to write
    """
    try:
        # Write user message
        file_handle.write(f"user: {turn['user']}\n")
        
        # Write expected response (can be multi-line)
        exp_text = turn.get('expected', '')
        if exp_text:
            # Split expected response into lines
            exp_lines = exp_text.split('\n')
            # Only the first line should have the "exp:" prefix
            if exp_lines:
                file_handle.write(f"exp: {exp_lines[0]}\n")
                # Subsequent lines don't have a prefix
                for line in exp_lines[1:]:
                    file_handle.write(f"{line}\n")
        
        # Write bot responses if they exist
        bot_responses = turn.get('bot', [])
        if isinstance(bot_responses, list):
            if bot_responses:
                # Write the first bot response with "bot:" prefix
                first_response = bot_responses[0]
                if '\n' in first_response:
                    # If the first response has newlines, handle those separately
                    resp_lines = first_response.split('\n')
                    file_handle.write(f"bot: {resp_lines[0]}\n")
                    for line in resp_lines[1:]:
                        file_handle.write(f"{line}\n")
                else:
                    file_handle.write(f"bot: {first_response}\n")
                
                # For remaining responses, write them without the prefix (as continuation lines)
                for resp in bot_responses[1:]:
                    if '\n' in resp:
                        resp_lines = resp.split('\n')
                        for line in resp_lines:
                            file_handle.write(f"{line}\n")
                    else:
                        file_handle.write(f"{resp}\n")
            else:
                # Include empty bot response line to indicate bot didn't respond
                file_handle.write("bot: \n")
        elif bot_responses:  # Handle case where bot_responses is a string
            # For consistency with expected responses, only first line gets bot: prefix
            if '\n' in bot_responses:
                resp_lines = bot_responses.split('\n')
                file_handle.write(f"bot: {resp_lines[0]}\n")
                # Subsequent lines don't have a prefix
                for line in resp_lines[1:]:
                    file_handle.write(f"{line}\n")
            else:
                file_handle.write(f"bot: {bot_responses}\n")
        else:
            # Include empty bot response line to indicate bot didn't respond
            file_handle.write("bot: \n")
        
        file_handle.write("---\n")
        file_handle.flush()  # Ensure content is written immediately
        
    except Exception as e:
        logger.error(f"Error writing turn to file: {e}")
        logger.error(traceback.format_exc())

def check_existing_simulations(test_file: str) -> bool:
    """Check if simulations already exist for a test file.
    
    Args:
        test_file: Path to the test file to check
    
    Returns:
        True if simulations exist, False otherwise
    """
    results_dir = Path("simulated_conversations")
    if not results_dir.exists():
        return False
        
    test_name = Path(test_file).stem
    existing_files = list(results_dir.glob(f"*_simulation_{test_name}_run*.txt"))
    
    if existing_files:
        logger.info(f"Found {len(existing_files)} existing simulation(s) for {test_file}")
        return True
    
    return False

async def run_conversation(turns: List[Dict[str, str]], config_path: str, turn_wait: int = 3, file_handle=None):
    """Run the conversation and capture bot responses."""
    
    try:
        # Handle directory (use base.py by default if it's a directory)
        config_path_obj = Path(config_path)
        if config_path_obj.is_dir():
            config_path = str(config_path_obj / "base.py")
            logger.info(f"Using base.py from directory: {config_path}")
        
        # Set environment variable for other modules to use
        os.environ["BOT_CONFIG_PATH"] = str(config_path)
        logger.info(f"Set BOT_CONFIG_PATH to {config_path}")
        
        # Import bot configuration
        spec = importlib.util.spec_from_file_location("bot_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Initialize bot client with config path
        try:
            # Import the BotClient class based on configuration
            bot_type = getattr(config_module, "BOT_TYPE", "botario_gpt")  # Default to GPT client
            logger.debug(f"Using bot type: {bot_type}")
            
            if bot_type == "botario_gpt":
                logger.debug("Using GPT-enabled Botario client")
                from api.bot_client_botario_gpt import BotClient
            else:
                logger.debug("Using standard Botario client")
                from api.bot_client_botario import BotClient
            
            logger.debug(f"Initializing BotClient with config_path={config_path}")
            client = BotClient(config_path=config_path)
            logger.debug("BotClient initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing bot client: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Run each turn
        for i, turn in enumerate(turns, 1):
            logger.debug(f"Running turn {i}/{len(turns)}")
            
            logger.info(f"User: {turn['user']}")
            logger.info(f"Expected: {turn['exp']}")
            
            # Get bot response
            try:
                response = await client.send_message(turn['user'])
                
                if response:
                    # Log raw text for debugging
                    logger.debug(f"Raw bot text: {response.text}")
                    logger.debug(f"Bot metadata: {response.metadata}")
                    
                    bot_responses = response.get_responses()
                    if bot_responses:
                        logger.info(f"Bot: {chr(10).join(bot_responses)}")
                    else:
                        logger.warning("Bot response was processed but contains no text")
                else:
                    logger.warning("No bot response object received")
                    bot_responses = []
            except Exception as e:
                logger.error(f"Error getting bot response: {str(e)}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                bot_responses = []
            
            # Write to file if handle provided
            if file_handle:
                try:
                    # Include both user message, expected response, and bot response(s)
                    write_turn_to_file(file_handle, {
                        'user': turn['user'],
                        'expected': turn['exp'],
                        'bot': bot_responses  # Include bot responses in the output file
                    })
                except Exception as e:
                    logger.error(f"Error writing turn to file: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    sys.exit(1)
            
            # Wait between turns
            if i < len(turns):
                logger.info(f"Waiting {turn_wait} seconds before next turn...")
                await asyncio.sleep(turn_wait)
        
        logger.info("Conversation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in conversation: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)  # Exit immediately on any error

async def run_single_simulation(test_file: str, config_path: str, run_number: int, turn_wait: int = 3, postfix: str = '', custom_results_dir: str = None):
    """Run a single simulation for a test file and save results.
    
    Args:
        test_file: Path to the test file to simulate
        config_path: Path to bot configuration file
        run_number: The run number for this simulation
        turn_wait: Time in seconds to wait between individual turns
        postfix: Optional postfix to add to the filename
        custom_results_dir: Optional custom directory to save results
    
    Returns:
        List of paths to saved simulation result files
    """
    try:
        # Read conversation script
        turns = read_conversation_script(test_file)
        if not turns:
            logger.error(f"No conversation turns found in {test_file}")
            return []
        
        # Import config module to get SIMULATIONS_DIR
        config_path_obj = Path(config_path).resolve()  # Convert to absolute path
        config_dir = config_path_obj
        
        if config_path_obj.is_dir():
            config_dir = config_path_obj
            config_path = str(config_path_obj / "base.py")
        else:
            config_dir = config_path_obj.parent
        
        # Set environment variable for other modules to use
        os.environ["BOT_CONFIG_PATH"] = str(config_path)
        logger.info(f"Set BOT_CONFIG_PATH environment variable to {config_path}")
        
        spec = None
        config_module = None
        try:
            spec = importlib.util.spec_from_file_location("sim_config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Resolve relative paths in the config module
            if hasattr(config_module, 'SIMULATIONS_DIR') and isinstance(config_module.SIMULATIONS_DIR, str):
                if config_module.SIMULATIONS_DIR.startswith('./'):
                    # Convert relative path to absolute
                    relative_path = config_module.SIMULATIONS_DIR[2:]  # Remove ./
                    config_module.SIMULATIONS_DIR = str(config_dir / relative_path)
                    logger.info(f"Resolved SIMULATIONS_DIR to absolute path: {config_module.SIMULATIONS_DIR}")
        except Exception as e:
            logger.warning(f"Could not load config module for SIMULATIONS_DIR: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
        # Create simulation file
        results_file, file_handle = create_simulation_file(test_file, run_number, postfix, custom_results_dir, config_module)
        if not file_handle:
            logger.error(f"Failed to create simulation file for {test_file}")
            return []
        
        # Run conversation
        success = await run_conversation(turns, config_path, turn_wait, file_handle)
        
        # Clean up
        file_handle.close()
        
        if not success:
            logger.error(f"Failed to run conversation for {test_file}")
            return []
            
        logger.info(f"Simulation completed successfully for {test_file}")
        return [results_file]
        
    except Exception as e:
        logger.error(f"Error in run_single_simulation: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

def shuffle_runs_without_consecutive_files(runs: List[Tuple[str, str, int]]) -> List[Tuple[str, str, int]]:
    """Shuffle runs in a way that prevents the same file from being run consecutively.
    
    Args:
        runs: List of (test_file, config_path, run_number) tuples
        
    Returns:
        Shuffled list of runs where the same file isn't run consecutively
    """
    if len(runs) <= 1:
        return runs
        
    # Group runs by test file
    runs_by_file = {}
    for run in runs:
        test_file, config_path, run_num = run
        if test_file not in runs_by_file:
            runs_by_file[test_file] = []
        runs_by_file[test_file].append((config_path, run_num))
    
    # Shuffle run numbers within each file
    for file in runs_by_file:
        random.shuffle(runs_by_file[file])
    
    # Get the list of files and shuffle it
    files = list(runs_by_file.keys())
    random.shuffle(files)
    
    # If we have only one file, just return shuffled runs for that file
    if len(files) == 1:
        return [(files[0], config_path, run_num) for config_path, run_num in runs_by_file[files[0]]]
    
    # Interleave runs from different files
    shuffled_runs = []
    file_index = 0
    last_file = None
    
    while any(runs_by_file[file] for file in runs_by_file):
        # Find the next file with remaining runs
        attempts = 0
        while (attempts < len(files) and 
               (not runs_by_file[files[file_index]] or files[file_index] == last_file)):
            file_index = (file_index + 1) % len(files)
            attempts += 1
            
        # If we're forced to use the same file twice in a row
        if attempts >= len(files) and runs_by_file[files[file_index]]:
            current_file = files[file_index]
            # Try to find ANY other file with runs
            for alt_file in files:
                if alt_file != last_file and runs_by_file[alt_file]:
                    current_file = alt_file
                    file_index = files.index(alt_file)
                    break
        else:
            current_file = files[file_index]
            
        if runs_by_file[current_file]:
            # Add a run from the current file
            config_path, run_num = runs_by_file[current_file].pop(0)
            shuffled_runs.append((current_file, config_path, run_num))
            last_file = current_file
            
        # Move to the next file
        file_index = (file_index + 1) % len(files)
    
    logger.info(f"Shuffled runs to avoid consecutive runs of the same file")
    return shuffled_runs

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run bot simulations.')
    parser.add_argument('--test', type=str, required=True, help='Path to test file or directory containing test files')
    parser.add_argument('--config', type=str, required=True, 
                      help='Path to configuration file or directory (if directory, uses base.py)')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs per test file')
    parser.add_argument('--turn-wait', type=int, default=3, help='Time in seconds to wait between individual turns')
    parser.add_argument('--postfix', type=str, default='', help='Postfix to add to the result filenames')
    parser.add_argument('--results-dir', type=str, required=False, 
                      help='Directory to save simulation results (overrides default project structure)')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Increase verbosity level (-v for INFO, -vv for DEBUG)')
    args = parser.parse_args()
    
    # Set up logging based on verbosity level
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    
    # Configure logging for different loggers
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set third-party loggers to WARNING
    for logger_name in ['asyncio', 'aiohttp', 'urllib3', 'httpx']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    try:
        # Get list of test files
        if os.path.isdir(args.test):
            test_files = glob.glob(os.path.join(args.test, '*.txt'))
        else:
            test_files = [args.test]
        
        if not test_files:
            logger.error(f"No test files found at {args.test}")
            sys.exit(1)
        
        # Create runs list
        runs = [(test_file, args.config, run_number) for test_file in test_files 
                for run_number in range(1, args.runs + 1)]
        
        # Shuffle runs to avoid consecutive runs of the same test
        runs = shuffle_runs_without_consecutive_files(runs)
        
        # Run simulations
        all_results = []
        total_runs = len(runs)
        
        for i, (test_file, config_path, run_number) in enumerate(runs, 1):
            logger.info(f"\nRunning simulation {i}/{total_runs} for {test_file} (run {run_number})")
            
            # Skip if simulation already exists
            if check_existing_simulations(test_file):
                logger.info(f"Skipping existing simulation for {test_file}")
                continue
            
            # Run simulation
            try:
                results = asyncio.run(run_single_simulation(
                    test_file=test_file,
                    config_path=config_path,
                    run_number=run_number,
                    turn_wait=args.turn_wait,
                    postfix=args.postfix,
                    custom_results_dir=args.results_dir
                ))
                all_results.extend(results)
                
                # Add small delay between simulations
                if i < total_runs:
                    time.sleep(random.uniform(0.5, 1.5))  # Random delay between 0.5 and 1.5 seconds
            except Exception as e:
                logger.error(f"Error in run_single_simulation: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                sys.exit(1)
        
        # Determine where the results were saved
        results_location = args.results_dir if args.results_dir else Path('simulated_conversations').resolve()
        logger.info(f"\nSimulation complete. Results saved in: {results_location}")
        if all_results:
            logger.info(f"Total simulations created: {len(all_results)}")
            logger.info(f"Simulation files: {', '.join(all_results)}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
