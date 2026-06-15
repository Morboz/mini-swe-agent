# Memory-Augmented SWE-Bench Agent Design

## Goal

Add an opt-in memory-augmented SWE-Bench execution path that:

- runs the existing SWE-Bench benchmark flow in Docker-backed environments,
- automatically ingests the current case repository into an external memory backend before the first model call,
- automatically performs one retrieval using the case problem statement,
- injects the retrieved markdown context into the conversation as a synthetic tool observation,
- aborts the current case if memory bootstrap fails,
- preserves existing `preds.json` and trajectory outputs so benchmark evaluation still works,
- records enough metadata to compare baseline and memory-augmented runs, especially token and cost consumption.

This design targets the current mini-SWE-agent codebase and the existing SWE-Bench runners:

- `src/minisweagent/run/benchmarks/swebench.py`
- `src/minisweagent/run/benchmarks/swebench_single.py`

## Non-Goals

- Exposing memory tools to the model for arbitrary later use
- Replacing the `bash` tool as the agent's execution mechanism
- Changing SWE-Bench evaluation format or `preds.json` schema
- Implementing multi-step retrieval or iterative search refinement
- Making memory bootstrap failures non-fatal

## Requirements

### Functional

1. Each SWE-Bench instance must automatically run a memory bootstrap before the first model query.
2. The bootstrap must execute exactly once per instance.
3. Bootstrap order is fixed:
   1. extract repository source files from the active environment,
   2. call memory `compile`,
   3. call memory `query`,
   4. inject retrieved context into the conversation.
4. The `query` input must be the SWE-Bench `problem_statement`.
5. The retrieved context is a markdown string and must be injected as a synthetic tool observation message.
6. If compile or query fails, the case must terminate immediately with a distinct exit status.
7. The standard benchmark outputs must still be produced:
   - trajectory JSON
   - `preds.json`

### Experimental

1. The memory path must be configurable so the same benchmark runner can be used for baseline and memory runs.
2. Runs must record memory bootstrap metadata and model token/cost usage for later comparison.
3. Baseline behavior must remain unchanged when memory is disabled.

## Proposed Approach

### Recommended approach: agent-level bootstrap

Implement memory bootstrap in a new agent class that wraps the first model query.

Why this is the best fit:

- It keeps retrieval augmentation in the agent layer, where message history is already managed.
- It avoids pushing prompt-construction concerns into the environment layer.
- It allows both `swebench.py` and `swebench_single.py` to adopt memory simply by switching `agent_class` or config.
- It makes A/B comparison straightforward because the runner and patch submission flow remain unchanged.

### Alternatives considered

#### Environment-level bootstrap

Compile/query would run as part of environment setup.

Rejected because:

- environment should own execution, not prompt augmentation,
- query results still need agent/model-specific injection,
- the boundary becomes harder to reason about and test.

#### Model wrapper bootstrap

Compile/query would run inside a model wrapper before the first API call.

Rejected because:

- the model layer should not need repository extraction or case-level orchestration,
- it couples model selection to benchmark/environment details,
- it makes token accounting and failure reporting less transparent.

## Architecture

### 1. Memory client

Add a dedicated client module for the external memory backend.

Responsibilities:

- call `/api/v1/compile`
- call `/api/v1/query`
- validate required response fields
- convert HTTP and schema issues into explicit internal exceptions
- measure request latency

Configuration:

- `base_url`
- `timeout_seconds`
- optional auth headers or bearer token
- `query_budget`
- `enable_w2`

Primary interface:

- `compile_repo(repo_id: str, files: list[SourceFilePayload], metadata: dict) -> CompileResponse`
- `query_repo(repo_id: str, query: str, metadata: dict, budget: int) -> QueryResponse`

### 2. Repository snapshot extractor

Add a utility that extracts repository files from the active environment before the first model query.

Responsibilities:

- determine the repository root inside the environment,
- enumerate candidate files,
- filter allowed source files,
- read file contents,
- construct memory backend `SourceFilePayload` items.

Filtering rules:

- include common source files such as:
  - `.py`, `.js`, `.ts`, `.tsx`, `.java`, `.go`, `.rb`, `.rs`, `.cpp`, `.c`, `.h`
- exclude directories:
  - `.git`
  - `node_modules`
  - `vendor`
  - `dist`
  - `build`
  - `.venv`
  - `__pycache__`
- exclude binary and obviously generated artifacts
- include test files, but mark them with `is_test=true`

Implementation detail:

Because the benchmark repository lives inside the execution environment, extraction should run through `env.execute(...)`, not host filesystem reads. The extractor should run a deterministic container-side script that:

- walks the repo,
- applies filtering,
- emits a JSON payload containing relative paths and file contents.

This avoids many individual `cat` calls and gives one stable integration point.

### 3. Memory bootstrap agent

Add a new agent class, tentatively `MemoryBootstrapAgent`, derived from the existing default interactive path used by SWE-Bench.

Responsibilities:

- track whether bootstrap has already run,
- perform bootstrap immediately before the first model query,
- inject retrieved context as a synthetic tool observation,
- terminate the case if bootstrap fails,
- record memory metadata in the saved trajectory.

Bootstrap sequence:

1. Extract repo source snapshot from environment.
2. Call memory compile with:
   - `repo_id = instance_id`
   - `files = extracted source files`
   - `metadata` including at least `instance_id` and basic runner metadata
3. Call memory query with:
   - `repo_id = instance_id`
   - `query = problem_statement`
4. Inject the returned markdown string into the message history before the first real model response is requested.

### 4. Synthetic observation injection

Retrieved memory context will be inserted as a synthetic observation, not concatenated into the system or user prompt.

Target shape:

- one assistant message that represents a synthetic tool call to `memory_search`, followed by
- one tool/result message containing the markdown response.

The injected content should be explicit that this is pre-fetched memory context. The tool-result body should preserve the markdown string returned by the backend with minimal decoration.

Rationale:

- matches the existing message-history pattern better than prompt concatenation,
- keeps provenance clear in trajectories,
- allows later inspection and token attribution,
- minimizes prompt-template churn.

## Detailed Data Flow

For each benchmark instance:

1. Runner loads dataset item and creates the environment.
2. Runner creates the configured agent.
3. Agent `run()` starts and builds the initial system and user messages.
4. Before the first model query, memory bootstrap runs once.
5. The extractor reads filtered repo files from the environment.
6. `compile` sends the repo snapshot to the backend.
7. `query` sends the SWE-Bench issue text to the backend.
8. Returned markdown context is injected into message history as synthetic tool output.
9. Normal agent loop begins.
10. Agent uses existing `bash` tool workflow to inspect, edit, test, and submit a patch.
11. Runner writes trajectory and `preds.json` as usual.

## Failure Handling

Memory bootstrap failures are fatal for the current case.

Distinct exit statuses:

- `MemoryExtractFailed`
- `MemoryCompileFailed`
- `MemoryQueryFailed`
- `MemoryInjectionFailed`
- `MemoryBootstrapInvalidResponse`

Failure behavior:

- no fallback to baseline behavior,
- no model query is issued after bootstrap failure,
- trajectory still saves the failure metadata,
- `preds.json` entry is written with empty `model_patch`, matching existing failure patterns.

This is intentional because the experiment is measuring the memory-enabled path itself, not graceful degradation.

## Configuration Design

Add an optional `memory` config section.

Suggested fields:

- `enabled: bool`
- `base_url: str`
- `timeout_seconds: int`
- `query_budget: int`
- `enable_w2: bool`
- `auth_token_env_var: str | None`
- `max_files: int | 0`
- `max_total_bytes: int | 0`
- `allowed_extensions: list[str]`
- `excluded_dirs: list[str]`

Behavior:

- when `memory.enabled` is false or absent, existing benchmark behavior is unchanged,
- when enabled, runner config selects the memory-aware agent class.

## Token and Cost Accounting

The current codebase tracks cost reliably, but not benchmark-level token summaries.

To support the experiment, add per-trajectory aggregation of model usage when present in provider responses.

Recorded metrics should include:

- `instance_cost`
- `api_calls`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

Rules:

- aggregation must be best-effort because providers differ in response shapes,
- missing usage values should be recorded as zero, not guessed,
- memory backend token usage is out of scope unless the backend returns explicit values and the experiment later chooses to track them separately.

This design intentionally keeps LLM token accounting separate from memory backend compute accounting.

## Output and Observability

Trajectory output should include:

- standard existing agent/model/environment info,
- memory bootstrap metadata,
- synthetic memory messages,
- aggregated token usage,
- failure reason if bootstrap fails.

Additional `info.memory` fields:

- `enabled`
- `repo_id`
- `compile_success`
- `query_success`
- `compile_latency_ms`
- `query_latency_ms`
- `source_file_count`
- `source_total_bytes`
- `context_chars`

## Testing Strategy

### Unit tests

1. Memory client request/response validation
2. Repo extractor filtering behavior
3. Repo extractor handling of excluded directories and file extensions
4. Agent bootstrap runs exactly once
5. Successful bootstrap injects synthetic messages before first model query
6. Compile failure aborts the run with the expected exit status
7. Query failure aborts the run with the expected exit status
8. Token aggregation works for known response shapes and tolerates missing usage

### Integration-style tests

1. `swebench.py` with memory disabled behaves exactly as before
2. `swebench.py` with memory enabled and mocked backend writes normal `preds.json`
3. `swebench_single.py` with memory enabled injects bootstrap context before solving
4. Failed memory bootstrap still writes a trajectory and an empty patch result

## Rollout Plan

Implement in three stages:

1. Add memory client and repo extractor behind tests.
2. Add memory bootstrap agent and wire it through benchmark config.
3. Add token aggregation and trajectory metadata for experiment analysis.

This staging keeps the core bootstrap path isolated from later reporting refinements.

## Open Decisions Resolved

- Memory bootstrap runs automatically once per instance: yes
- Query runs once at task start only: yes
- Query result injection format: synthetic tool observation
- Bootstrap failure policy: fail the case immediately
- Repo ingest scope: filtered source files plus tests, with tests flagged
- Query response format: markdown string, preserved as-is

## Scope Check

This scope is focused enough for a single implementation plan:

- one new client area,
- one new extraction utility,
- one new agent variant,
- limited benchmark wiring,
- bounded reporting additions.

It does not require redesigning the benchmark runner or model abstraction.
