# File Structure

This document describes the layout of the project's directory structure.

## Project Layout

- **`/`**: The root directory.
  - **`.gitignore`**: Specifies intentionally untracked files to ignore.
  - **`README.md`**: The main README file for the project.
  - **`requirements.txt`**: A list of the python dependencies for the project.
  - **`.instructions/`**: Contains all the documentation and guidelines for the project.
  - **`.issue/`**: Contains issue-specific folders for planning and research.
  - **`projects/`**: Contains all the different projects.
    - **`[project_name]/`**: The project specific folder.
      - **`cases/`**: Test cases for the project.
      - **`config/`**: Configuration files for the project.
      - **`results/`**: Results of test runs.
      - **`simulated_conversations/`**: Simulated conversations from test runs.
  - **`src/`**: Contains the source code for the project.
    - **`api/`**: API clients and related modules.
    - **`conversation/`**: Conversation handling logic.
    - **`metrics/`**: Metrics and evaluation logic.
    - **`scripts/`**: Scripts for running simulations, evaluations, etc.
    - **`utils/`**: Utility functions and helpers.

