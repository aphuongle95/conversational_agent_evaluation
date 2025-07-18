# Testing Guidelines

This document provides guidelines for writing and running tests in this project.

## General Principles

- **Write tests for all new features and bug fixes.**
- **Tests should be independent and repeatable.**
- **Keep tests focused on a single piece of functionality.**

## Running Tests

To run the tests, use the following command:

```bash
pytest
```

## Test Structure

- **Unit Tests:** Test individual functions and classes in isolation.
- **Integration Tests:** Test the interaction between multiple components.
- **End-to-End (E2E) Tests:** Test the entire application from start to finish.

## Test Naming

- Test files should be named `test_*.py` or `*_test.py`.
- Test functions should be named `test_*`.
