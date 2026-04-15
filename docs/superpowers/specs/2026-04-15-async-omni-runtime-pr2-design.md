# Async Omni Runtime PR2 Design

## Goal

Finish the next runtime ownership step after PR1:

- keep the public `AsyncOmniEngine` API stable,
- keep the current `janus` dict-message envelope compatible,
- move stage-0 runtime ownership out of the caller thread and into the background runtime,
- migrate CFG companion stage-0 handling onto the same background ingress path,
- make `PipelineRuntime` the explicit multi-stage control-plane boundary.

This PR does **not** attempt the full bootstrap extraction yet. It only removes caller-thread stage-0 runtime behavior.

## Why PR2 Exists

PR1 introduced:

- `PipelineRequestState`,
- `StageRuntime`,
- a runtime-backed orchestrator loop,
- an explicit temporary compatibility boundary in `AsyncOmniEngine`.

That boundary is still the main architectural mismatch:

- `AsyncOmniEngine.add_request()` still preprocesses stage-0 LLM requests,
- caller thread still performs stage-0 output registration,
- CFG companion expansion and stage-0 preparation still live outside the background runtime.

PR2 removes that mismatch without changing the external API.

## Scope

### In Scope

- explicit `PipelineRuntime` extraction,
- background ownership of stage-0 external request ingress,
- background ownership of stage-0 streaming-update ingress,
- background ownership of CFG companion expansion and stage-0 ingress,
- shared stage-0 preprocessing helpers for LLM entry stages,
- preserving current output routing, abort behavior, async-chunk prewarm, and janus queue usage.

### Out of Scope

- typed queue commands/events,
- full `stage_bootstrap.py` extraction,
- transport rewrites,
- topology graph generalization,
- public API redesign,
- per-request stream handle redesign.

## Architecture

### Target Layering for PR2

```text
AsyncOmniEngine
  -> PipelineRuntime
      -> entry_runtime
      -> stage_runtimes[*]
```

### Responsibility Split

#### `AsyncOmniEngine`

`AsyncOmniEngine` should be a thin facade:

- accept caller requests,
- build compatible janus messages,
- enqueue those messages,
- read outputs,
- bootstrap the background runtime thread,
- own top-level shutdown.

It should no longer:

- call `input_processor.process_inputs()` for stage-0 work,
- register stage-0 requests with `output_processors[0]`,
- expand CFG companions into preprocessed stage-0 requests.

#### `PipelineRuntime`

`PipelineRuntime` is the multi-stage control plane:

- owns request graph and `PipelineRequestState`,
- owns companion graph and deferred-parent coordination,
- owns stage routing, metrics, abort fanout, and terminal output emission,
- resolves the entry runtime for external ingress,
- triggers CFG companion expansion in the background runtime.

`PipelineRuntime` decides **where/when** work enters the pipeline.

#### `StageRuntime`

`StageRuntime` remains the stage-local execution boundary:

- stage-local preprocessing,
- output-processor registration,
- stage-local submit,
- stage-local poll/process,
- stage-local abort,
- stage-local shutdown.

The entry runtime decides **how** external ingress is accepted for its stage.

## Data Model

PR2 adopts a typed request-state model, but keeps it intentionally narrow.

### `RequestMeta`

Stable request control-plane metadata:

- `request_id`
- `final_stage_id`
- `sampling_params_list`
- `prompt_text`
- `arrival_time`
- `lora_request`
- `tokenization_kwargs`
- `trace_headers`
- `priority`
- `data_parallel_rank`
- `reasoning_ended`
- `resumable`

### `PipelineData`

Evolving pipeline working state:

- `raw_prompt`
- `stage0_request`
- `terminal_outputs`

PR2 must not introduce a generic `dict[str, Any]` payload bucket. Add only the fields actually needed for this slice.

### `PipelineRequestState`

`PipelineRequestState` should become:

- pipeline-owned lifecycle state,
- plus `meta`,
- plus `data`,
- plus active stage bookkeeping.

Companion graph state remains pipeline-owned rather than being folded into `PipelineData` in this PR.

## Entry Runtime Design

PR2 must avoid magic-number semantics like `StageRuntime[0]`.

`PipelineRuntime` should resolve:

- `entry_stage_id`
- `entry_runtime`

External ingress flows use `entry_runtime`, not `stage_runtimes[0]`.

This keeps the implementation aligned with topology role rather than array position, while still preserving the current list-based pipeline semantics.

## Operation-Oriented Stage API

PR2 should not introduce a giant transport-shaped object like `Stage0IngressRequest`.

Instead, `StageRuntime` gains explicit ingress operations:

- `accept_external_request(meta: RequestMeta, data: PipelineData) -> Any`
- `accept_streaming_update(meta: RequestMeta, data: PipelineData) -> Any`

The existing `submit(...)` remains the API for already-prepared stage-local requests used by downstream forwarding.

CFG companions reuse `accept_external_request(...)`; they do not get a third ingress API.

## Stage-0 Processing Helpers

PR2 introduces a small shared helper module:

- `vllm_omni/engine/stage0_processing.py`

This module centralizes the current stage-0 LLM preparation details:

- global request-id injection,
- `input_processor.process_inputs(...)`,
- omni request upgrade,
- `external_req_id` repair,
- final-stage metadata attachment,
- stage-0 output registration.

This module exists to share implementation, not to define a new runtime abstraction.

## Stage Behavior

### `LLMStageRuntime`

For entry ingress:

- `accept_external_request(...)` performs full stage-0 preprocess/register/submit,
- `accept_streaming_update(...)` performs full stage-0 streaming preprocess/register/submit.

For forwarded requests:

- `submit(...)` still accepts already-prepared stage-local requests.

### `DiffusionStageRuntime`

For entry ingress:

- `accept_external_request(...)` is raw pass-through submit,
- `accept_streaming_update(...)` is raw pass-through submit.

No LLM-style preprocessing is introduced for diffusion entry stages.

## Message Compatibility

PR2 keeps the janus dict-message protocol compatible at the envelope level.

The following keys stay stable:

- `type`
- `request_id`
- `prompt`
- `original_prompt`
- `sampling_params_list`
- `final_stage_id`

PR2 may add more metadata fields to support background ingress, but must not break existing envelope expectations.

Important semantic change:

- for stage-0 external ingress, `msg["prompt"]` returns to meaning the raw caller prompt,
- not a caller-thread-preprocessed `OmniEngineCoreRequest`.

That semantic change is internal to the background runtime boundary and is the intended PR2 shift.

## PipelineRuntime Extraction

PR2 introduces:

- `vllm_omni/engine/pipeline_runtime.py`

`orchestrator.py` should become a compatibility wrapper or alias so existing imports do not break during this slice.

The extracted `PipelineRuntime` owns:

- `_handle_add_request`
- `_handle_streaming_update`
- `_handle_add_companion`
- companion expansion trigger
- output routing
- kv-ready routing
- abort fanout
- collective RPC
- shutdown coordination

## CFG Companion Migration

PR2 moves CFG companion expansion out of `AsyncOmniEngine`.

New flow:

1. external request enters `PipelineRuntime`,
2. `PipelineRuntime` creates parent request state,
3. `PipelineRuntime` expands companions using `prompt_expand_func`,
4. parent and companions each enter through `entry_runtime.accept_external_request(...)`,
5. companion graph remains pipeline-owned,
6. deferred-parent forwarding and companion-ready logic remain unchanged in meaning.

This ensures there is only one stage-0 ingress path.

## Migration Order

### Task A: Background Entry Ownership

Implement:

- `RequestMeta`
- `PipelineData`
- `stage0_processing.py`
- entry-runtime ingress APIs
- `PipelineRuntime` extraction
- background handling of normal requests and streaming updates

Must preserve:

- async-chunk prewarm,
- output routing,
- `external_req_id == global request id` compatibility.

### Task B: CFG Companion Migration

Implement:

- background companion expansion,
- companion reuse of entry ingress,
- removal of caller-thread companion preprocessing/registration paths from `AsyncOmniEngine`.

Must preserve:

- companion graph correctness,
- deferred-parent behavior,
- abort cleanup parity for parent + companions.

## Test Strategy

### Tests That Must Flip Semantics

#### `tests/engine/test_async_omni_engine_input.py`

Must flip from proving:

- caller-thread preprocessing and stage-0 registration happen locally

to proving:

- `AsyncOmniEngine` builds raw ingress messages,
- caller thread does not preprocess stage-0 requests,
- caller thread does not register stage-0 requests locally.

#### `tests/engine/test_orchestrator.py`

Must add coverage that:

- raw external prompt enters the background runtime,
- entry runtime performs stage-0 ingress preparation,
- non-dense `stage_id` regression remains covered.

#### `tests/engine/test_stage_runtime.py`

Must add direct unit coverage for:

- `LLMStageRuntime.accept_external_request(...)`,
- `LLMStageRuntime.accept_streaming_update(...)`,
- diffusion pass-through entry ingress behavior.

#### CFG Companion Coverage

Must prove that:

- companion expansion is background-owned,
- companion preprocessing/registration is background-owned,
- parent/companion lifecycle and abort behavior remain correct.

## Risks

### Main Risk

Moving stage-0 registration off the caller thread changes when request-local output-processor state appears.

This is why PR2 must add direct stage-runtime ingress tests rather than relying only on integration coverage.

### Secondary Risk

Streaming-update behavior can accidentally diverge from normal add-request ingress if they do not share the same stage-local preparation contract.

PR2 should keep these paths parallel in interface and test shape.

## Success Criteria

PR2 is successful when all of the following are true:

1. `AsyncOmniEngine` no longer performs stage-0 preprocessing in the caller thread.
2. `AsyncOmniEngine` no longer performs stage-0 output registration in the caller thread.
3. `PipelineRuntime` is the explicit multi-stage control-plane boundary.
4. `PipelineRuntime` resolves and uses an `entry_runtime`, not `stage_runtimes[0]`.
5. Normal requests, streaming updates, and CFG companions all enter through the same background ingress model.
6. `submit(...)` remains the stage-local API for forwarded downstream requests.
7. Public API and janus message envelope remain compatible.
8. Full bootstrap extraction is still deferred to the next slice.

## Deferred Follow-Ups

Not part of PR2:

- `stage_bootstrap.py`
- typed commands/events
- explicit topology graph
- richer side-channel abstractions
- request-handle / per-request stream redesign

Those are follow-on slices after stage-0 ownership is cleanly inside the background runtime.
