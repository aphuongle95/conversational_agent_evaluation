# Conversational Agent Evaluation Framework

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A robust, automated framework for evaluating and testing conversational AI systems through simulation, evaluation, and comprehensive metrics analysis. This project enables systematic testing of chatbots and conversational agents with state-of-the-art techniques including RAGAs answer accuracy, semantic similarity, and multi-turn conversation validation.

## Key Features

- **End-to-End Testing**: Automated conversation simulation and evaluation
- **Advanced Metrics**: Includes RAGAS for answer accuracy assessment
- **Reproducible Results**: Multiple test runs for consistency validation
- **Configurable**: Supports various conversation agents through modular APIs
- **Comprehensive Reporting**: Detailed metrics and visualization options

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy the example environment file and configure:
```bash
cp .env.example .env
# Edit .env with your configuration
```
## Generating Test Cases

There are three main ways to generate test cases:

### From a Single Conversation ID

Generate test cases from a specific conversation:
```bash
python src/scripts/cases_generation.py --conversation-id "<CONVERSATION_ID>" --output cases/test_id_description.txt 
```

### From a Dialog ID

Generate test cases from a specific dialog:
```bash
python src/scripts/cases_generation.py --mode conversation --dialog-id "<DIALOG_ID>" --output cases/test_dialog.txt
```

### From a JSON File (Batch Mode)

Generate multiple test cases from a JSON file:
```bash
python src/scripts/cases_generation.py --json-file example_dialogs.json --output-dir cases
```

#### JSON Format

The JSON file should follow this structure:
```json
{
  "auth_token": "your_authentication_token",
  "dialogs": [
    "dialog_id_1",
    "dialog_id_2",
    "dialog_id_3"
  ]
}
```

This format efficiently processes multiple dialogs with a single authentication token.

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

```
.
├── src/                       # Source code
│   ├── api/                   # API clients for different bot platforms
│   ├── config/                # Configuration settings
│   ├── conversation/          # Conversation handling logic
│   ├── metrics/               # Evaluation metrics implementation
│   │   ├── llm_metrics.py     # LLM-based metrics
│   │   └── ragas_metrics.py   # RAGAS evaluation metrics
│   ├── scripts/               # Main executable scripts
│   │   ├── simulation.py      # Run conversation simulations
│   │   ├── evaluation.py      # Evaluate simulated conversations
│   │   └── cases_generation.py # Generate test cases
│   └── utils/                 # Utility functions and helpers
├── projects/                  # Project-specific configurations and cases
│   └── [project_name]/        # Organization by project
│       ├── cases/             # Test cases for the project
│       ├── config/            # Configuration files
│       ├── results/           # Evaluation results
│       └── simulated_conversations/ # Simulation results
├── .instructions/             # Development standards and guidelines
├── requirements.txt           # Project dependencies
└── README.md                  # This documentation
```

## Features

- **Comprehensive Testing Framework**:
  - Separate simulation and evaluation phases
  - Multiple simulation runs for consistency testing
  - Stateful conversation tracking

- **Flexible Integration**:
  - API-based bot interaction
  - Support for multiple bot platforms
  - Configurable testing parameters

- **Advanced Evaluation**:
  - Multi-turn conversation testing
  - Multiple comparison metrics
  - RAGAs answer accuracy metrics
  - Semantic similarity scoring

- **Detailed Analytics**:
  - Per-conversation reporting
  - Aggregate statistics across runs
  - Failure analysis and visualization
  - Configurable metrics thresholds

## Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -am 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Submit a pull request

Please make sure to update tests as appropriate and adhere to the project's coding standards.

## Security Best Practices

To maintain the security of your application:

1. **Environment Variables**: Store all sensitive credentials in your `.env` file and never commit this file to version control
2. **Credential Rotation**: Regularly rotate API keys and tokens
3. **Access Controls**: Use the principle of least privilege when configuring API permissions
4. **Parameter Validation**: All user inputs are validated before processing
5. **Request Limiting**: The framework includes rate limiting to prevent API abuse

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.