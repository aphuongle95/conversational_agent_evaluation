# Bot Testing Framework

Automated testing framework for evaluating bot conversations using Ragas answer accuracy metric.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
AI_GATEWAY_TOKEN=your_token
AI_GATEWAY_CHAT_ENDPOINT=your_endpoint
EVAL_PROFILE=your_profile

```
## Generating Test Cases

There are three main ways to generate test cases:

### From a Single Conversation ID

Generate test cases from a specific Botario conversation:
```bash
python src/scripts/cases_generation.py --conversation-id "9900a9d9-edbe-4ea0-928b-b8c2db497169" --output cases/test_id_description.txt 
```

### From a Cognitive Voice Dialog ID

Generate test cases from a specific Cognitive Voice dialog:
```bash
python src/scripts/cases_generation.py --mode conversation --reseller-token "3d321dff-56f0-4f25-83e5-ff74de2238fa" --dialog-id "bbc72b6b-e160-48b8-ba38-dd0cce7d940b" --output cases/cases_real/test_bbc72b6b-e160-48b8-ba38-dd0cce7d940b.txt
```

### From a JSON File (Batch Mode)

Generate multiple test cases from a JSON file:
```bash
python src/scripts/cases_generation.py --json-file example_dialogs.json --output-dir cases
```

#### JSON Format

The JSON file should follow this structure with a single reseller token for multiple dialogs:
```json
{
  "reseller_token": "your_reseller_token_here",
  "dialogs": [
    "dialog_id_1",
    "dialog_id_2",
    "dialog_id_3"
  ]
}
```

This format efficiently processes multiple dialogs with a single reseller token.

### Update Test Cases with Bot Responses

Use the simulation logic to interact with the bot and update predefined test cases with actual bot responses:
```bash
python src/scripts/cases_generation.py --mode interaction --test cases/test_simple.txt --output-dir processed_cases
```

You can also process multiple files or directories:
```bash
python src/scripts/cases_generation.py --mode interaction --test cases/feedback_table/test_* cases/regression/test_xyz.txt --output-dir processed_cases
```

Parameters:
- `--mode interaction`: Required to enable interaction mode
- `--test`: One or more test file paths or directories to process
- `--output-dir`: Directory to save processed files (default: cases)
- `--run-wait`: Wait time in seconds between files to avoid rate limits (default: 30)

This mode uses the same simulation logic as the testing process to:
1. Read the predefined test cases
2. Run each conversation with the bot
3. Capture the actual bot responses
4. Update the test cases by replacing the expected responses with the actual bot responses

### Conversation Turn Extraction

The test case generator automatically:

1. Combines consecutive messages from the same speaker (user or bot) into a single line
2. Joins multiple messages with spaces to create coherent, readable sentences
3. Preserves conversation flow, maintaining the user-bot turn structure
4. For Cognitive Voice dialogs, specially handles initial system greetings
5. Saves the conversation in a format ready for simulation testing

## Running Tests

The testing process consists of two steps:

### 1. Run Simulation

First, simulate the conversations. You can:

Run a specific test file:
```bash
python src/scripts/simulation.py --test cases/test_simple.txt --num-runs 3
```

Run all test files matching a pattern:
```bash
python src/scripts/simulation.py --test cases/feedback_table/test_* --num-runs 3
```

Run multiple specific test files or patterns:
```bash
python src/scripts/simulation.py --test cases/feedback_table/test_* cases/regression/test_xyz.txt --num-runs 3
```

Or run all test files in the cases directory:
```bash
python src/scripts/simulation.py --num-runs 3
```

Parameters:
- `--test`: One or more test file paths or glob patterns
- `--num-runs`: Number of times to run each test case (default: 3)
- `--run-wait`: Wait time in seconds between runs to avoid rate limits (default: 30)

This will generate simulation results in the `simulated_conversations` directory.

### 2. Run Evaluation

Then, evaluate the simulated conversations:
```bash
python src/scripts/evaluation.py --simulations simulated_conversations/simulation_*.txt
```

Parameters:
- `--simulations`: Paths to simulation result files (required, supports wildcards)
- `--skip-turns`: Number of initial turns to skip in evaluation (default: 0)

## Test Files

Test files should be placed in the `cases` directory with format:
```
user: user message
exp: expected bot response
---
user: next user message
exp: next expected response
```

## Project Structure

- `src/`
  - `api/` - Bot API client
  - `config/` - Configuration settings
  - `conversation/` - Conversation test handler
  - `metrics/` - evaluation metrics
  - `scripts/` - Utility scripts
    - `simulation.py` - Run conversation simulations
    - `evaluation.py` - Evaluate simulated conversations
    - `cases_generation.py` - Generate test cases from conversation data
- `cases/` - Test conversation files
- `simulated_conversations/` - Simulation results
- `results/` - Evaluation results (gitignored)

## Features

- Separate simulation and evaluation phases
- Multiple simulation runs for consistency testing
- API-based bot interaction
- Multi-turn conversation testing
- Multiple comparison metrics for output validation
- Configurable testing parameters
- Detailed test reporting
- Aggregate statistics across multiple runs