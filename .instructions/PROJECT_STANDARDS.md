# Project Guidance & Standards

This document outlines the project workflow, planning process, and development standards that all contributors must follow.

## Development Workflow

### Phase 1: Prepare your git branch

- Create and switch to a branch named `fix-issue-[id]` where `[id]` is the GitHub issue number.
- Check for an existing `.issue/[id]` folder. If it exists, resume work from there.
- If it doesn't exist, create `.issue/[id]` folder and a `plan.json` file inside it.

### Phase 2: Create a step-by-step plan

Create a numbered step-by-step plan in `plan.json`. When creating the plan:

1. **Practice Test-Driven Development (TDD):**
   - For any new functionality or bug fix, first create a failing test.
   - After implementing the code, update or add tests to ensure coverage.
   - Only run tests for the files you modify (no preflight tests).
   - Follow Python testing conventions as described in `TESTING_GUIDELINES.md`.

2. **Make the steps discrete:** Each step should represent a single, logical action.

3. **Optimize for clarity and detail:** Ensure steps are descriptive and unambiguous.

4. **Always finish with a pull request** merging into the main branch.

Each step in `plan.json` must contain:

- **step:** (Integer) Incremental step number starting at 1.
- **prompt:** (String) A detailed description of the work to be done.
- **status:** (String) Current status ("pending", "completed", or "failed").
- **time:** (String) ISO 8601 timestamp updated upon completion.
- **git:** (Object) Git-related information:
  - **commit_message:** (String) Format: "[Step X] <description>"
  - **commit_hash:** (String) Full git commit hash after step completion.

### Phase 3: Request Approval

Before executing any steps, present the plan for approval.

### Phase 4: Step-by-Step Execution

After approval, execute the plan sequentially:

1. Announce each step before execution.
2. If a step fails, try to resolve it or ask for help.
3. For no-code-change steps, commit with `--allow-empty`.
4. After each step:
   - Stage and commit changes
   - Update `plan.json` with status, timestamp, and commit hash
   - Document actions in `.issue/[id]/research/`
5. Only proceed after each step completes successfully.
6. Do not delete the `plan.json` file after completion.
