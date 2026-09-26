# Quality, Timing and Serving Contract

> Planned protocol. No latency target or capability is claimed as achieved.

## Output and quality

Raw ASR text is always available. A separate conservative editor may format it
only under [the editor contract](EDITOR_AND_RL_CONTRACT.md). Rejected, malformed,
timed-out or uncertain edits return the original transcript with reason/status.
Provisional tokens are visibly provisional; only validated completed output can
be inserted as final text. Spoken instructions within the transcript are data.

ASR quality: WER/CER with a frozen normalization rule, names/numbers/negation
errors and clean-speech regression. Confidence: defined for the exact
checkpoint/head/decoder/precision, fitted on calibration data, checked with
reliability/ECE/Brier and risk-coverage. Selecting a threshold is not calibration.
Refit/revalidate after inference or checkpoint changes; unavailable beam-word
alignment must not inherit greedy confidence scores.

Editor quality: proposed and delivered lexical/protected-span violation rates,
formatting accuracy, identity/needs-edit performance, edit coverage, fallback
rate, independent human meaning review. An LLM may preserve an ASR error; it
cannot recover absent acoustic evidence from text alone. No personalization
claim without a separate user-specific protocol, which this release excludes.

## Timing boundaries

Every trace uses request_id, stream_id, sequence numbers, stage start/end events,
checkpoint/config IDs, hardware/topology, load and input/output lengths. Use
monotonic clocks within a process. Cross-host durations require explicit clock
synchronization or client round-trip measurement; do not subtract unrelated clocks.

| Metric | Start | End |
| :--- | :--- | :--- |
| Time to first partial | First audio sample received by service | First nonempty partial received by client |
| Post-utterance ASR finalization | Annotated speech-end time on replay clock | Final ASR transcript received by client |
| Server LLM TTFT | LLM request enqueued after final transcript | First generated token available at server |
| Client editor TTFT | Editor request sent by client | First provisional token received by client |
| LLM completion | Same server enqueue event | EOS or declared generation stop |
| Final usable text | Annotated speech-end time on replay clock | Guard-validated final text or flagged fallback received by client |

Speech-end annotation is offline evaluation metadata, not available to the live
endpoint detector. Also report detector-to-final intervals when no human speech
end exists, labeled as that proxy. Fixed-rate prerecorded replay is measured
system execution, but is not the same as a live microphone trial. Audio duration
is not silently subtracted from capture-to-final wall time.

Queue, endpoint wait, preprocessing, ASR GPU compute, decoding, editor prefill,
generation, guard, serialization and client/network spans are recorded separately.
TTFT is part of completion, not another additive stage. Overlapping ASR/audio
capture and asynchronous stages are represented as intervals with parent links.
Compute end-to-end duration from actual start/end events, not a sum of overlapping
work. Do not add p95/p99 stage values or assign the tail owner from whichever
independent stage has the largest percentile. Inspect the same slow request IDs,
their critical paths and a controlled intervention before attributing causality.

## Workload, targets and fair comparisons

Day 15 establishes the default bounded editor workload: <=256 input tokens,
<=128 generated tokens, fixed prompt template, warm batch-1 concurrency-1,
one L4 and documented host CPU. A **warm server LLM TTFT p95 below 500 ms** is
an experimental target on that workload, not an end-to-end requirement or a
guarantee. Report misses honestly alongside quality; changing lengths, load,
hardware or topology creates another condition, not an improved same-condition run.

Record cold startup/model-load/compile time separately, retain all valid slow
requests, and document only instrumentation failures as exclusions. Prefix-cache
hit/miss is measured only if the selected runtime supports and exposes it;
otherwise mark unsupported. Cache reuse does not remove decode work or network
time. Compiler/graph/batch/precision techniques are hypotheses: choose by measured
profile, not a mandatory checklist of every runtime feature.

Use >=100 repeated requests per primary condition for median/p95 diagnostics;
p99 requires >=1000 completed requests per reported condition or must be labeled
exploratory/insufficient. Always show counts, seeds, repeat/run variation, failures,
length slices and cold/warm conditions. Confidence intervals for quality should
resample by original utterance/group; repeated corruptions are not independent
speakers. The frozen >=30-utterance set is a small diagnostic benchmark.

## Resource topology and final-stack validation

Default pilot/deployment topology: one L4 worker, one ASR checkpoint and one small
editor loaded concurrently, CPU VAD/decoding, bounded per-stream state, and a
serialized GPU work queue before advanced batching. At the early baseline measure
combined peak memory and ASR interference during editor generation, not the sum
of isolated models' claimed footprints. Training unloads serving models.

If co-residency fails, record the blocker and request a revised topology/budget;
do not silently add a GPU, hot-swap model loads out of timing, or fabricate joint
capacity. The final selected ASR and editor checkpoints must rerun compatibility,
calibration, combined memory and load tests after training. Early optimization
tables are provisional until this final-stack revalidation.

Load testing increases 1, 2, 4, 8... streams only until the authorized memory,
duration/spend or quality/latency limit. Report offered and achieved rates,
per-stream fairness, queue wait, rejected/cancelled/failed/timed-out work and
successful response latency. Rejection cannot masquerade as higher capacity.
Bound queue depth and waiting time, clean up session state, apply cancellation
between safe work units, and measure residual GPU work instead of promising
instant kernel preemption. Reproduce overload, disconnect and recovery.

Cost = measured billed compute duration times recorded dated rates, with idle,
startup and unsuccessful requests included. Normalize by **successfully processed**
audio duration/request count. Cost per 1000 audio hours is an extrapolation with
utilization assumptions, not evidence of serving that volume or millions of users.

## Primary metric reference

```text
https://docs.vllm.ai/en/latest/design/metrics/
```

Terminology checked September 26, 2026. vLLM is a reference, not a required or
installed backend; use equivalent instrumented clocks in the selected runtime.