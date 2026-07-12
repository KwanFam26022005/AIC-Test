# Caption Pipeline with Audio Context - Detailed Design

## 1. Document status

- Status: design only; this document does not implement code.
- First validation target: L22_V012.
- Deployment target: one or two RTX A5000 24 GB GPUs.
- Caption language: English.
- Audio language: preserve the upstream transcript language, currently Vietnamese.
- Baseline visual model: Qwen/Qwen2.5-VL-7B-Instruct.
- Shot model: configurable between Qwen2.5-7B-Instruct and Qwen2.5-3B-Instruct.

This document is the implementation contract for the new caption branch. The
audio pipeline already exists and is treated as a read-only upstream dependency.

## 2. Scope

### 2.1 In scope

1. Load keyframes, timestamps, and shot membership.
2. Select one anchor and optional visually novel frames per shot.
3. Generate visual-only frame captions.
4. Propagate captions to redundant frames with explicit provenance.
5. Read and align existing ASR segments to shots.
6. Generate shot caption, event caption, TRAKE fields, and ReCap memory in one
   text-model request per shot.
7. Validate, checkpoint, resume, and report all caption outputs.
8. Reserve a stable contract for optional multi-image LVLM escalation.

### 2.2 Out of scope

- Audio extraction, VAD, ASR inference, or audio environment changes.
- OCR and object-detection inputs.
- OCR/object evidence inside prompts or caption schemas.
- Elasticsearch mappings, embedding generation, RRF, or cross-modal index fusion.
- Retrieval evaluation and visualization UI.

The caption branch may reference ASR feature IDs and aligned transcript text,
but it must not rewrite the upstream audio output.

## 3. Core decisions

### 3.1 Evidence separation

Frame captions answer:

~~~text
What is visibly present in this frame?
~~~

Shot captions answer:

~~~text
What happens in this shot, considering visual evidence and aligned speech?
~~~

Visual observations and spoken context remain separate fields. Audio must not
silently become a claim about what is visible.

### 3.2 Late audio context

ASR is not passed into normal frame captioning. It is introduced only at shot
ReCap. This avoids repeating one transcript across multiple frames and reduces
audio-to-visual hallucination.

### 3.3 No frame-level text LLM pass

The Qwen2.5-VL output is the canonical generated caption for the selected frame.
There is no second text-LLM call for every keyframe.

### 3.4 One normal generation per shot

One structured shot request returns:

~~~text
shot_caption
temporal_caption
event_caption
action_state
temporal_role
actors
actions
objects_involved
scene
trake_text
memory_after
~~~

Shot captioning and TRAKE extraction are not separate model passes.

### 3.5 Caption reuse is auditable

A non-selected frame may inherit a caption only when frame selection marks it as
visually redundant. The result records its source frame, similarity method, and
score. It must never look like direct VLM inference on the target image.

### 3.6 Escalation is optional for the first milestone

The first L22_V012 implementation completes the optimized split path before
multi-image escalation is enabled. Escalation fields are included now so adding
it later does not require a schema migration.

## 4. Pipeline overview

~~~text
Existing inputs
├── keyframes + keyframe map
├── shot map or timestamp-gap shot grouping
└── audio_features.jsonl
          │
          ▼
Phase 0 - Preflight and canonical timeline
          │
          ▼
Phase 1 - Shot-aware frame selection
          ├── anchor/novel frames
          └── redundant frames + propagation source
          │
          ▼
Phase 2 - Qwen2.5-VL visual-only captioning
          │
          ▼
Phase 3 - Validation and caption propagation
          │
          ▼
Phase 4 - Read-only ASR alignment by shot
          │
          ▼
Phase 5 - Text ReCap, one structured request per shot
          │
          ▼
Phase 6 - Optional LVLM escalation/refinement
          │
          ▼
Phase 7 - Canonical outputs, checkpoints, and reports
~~~

## 5. Input contracts

### 5.1 Keyframes

Every frame requires:

| Field | Type | Requirement |
|---|---|---|
| video_id | string | L22_V012 for the first test |
| canonical_frame_id | string | Unique stable key |
| keyframe_idx | integer | Monotonic keyframe order |
| source_frame_idx | integer or null | Original video frame index when available |
| timestamp_sec | float | Required, non-negative, monotonic |
| image_path or image_relpath | string | Must resolve before GPU model load |

The L22_V012 baseline contains 283 keyframes from 0.0 to approximately 1199.93
seconds.

### 5.2 Shots

The first implementation may reuse the current timestamp-gap grouping. Each shot
requires:

~~~text
video_id
shot_id
start_sec
end_sec
ordered frame_ids
~~~

The current L22_V012 baseline contains 102 shots. This is a test acceptance
reference, not a global constant.

### 5.3 Existing audio features

Input:

~~~text
outputs/audio/L22_V012/features/audio_features.jsonl
~~~

Expected schema: audio_feature_v1.

Only rows with usable_for_caption=true are eligible. The caption branch reads:

| Field | Type | Purpose |
|---|---|---|
| feature_id | string | Provenance and deduplication |
| video_id | string | Video filter |
| start_sec | float | Temporal alignment |
| end_sec | float | Temporal alignment |
| caption_text | string | Preferred context |
| clean_transcript | string | Fallback text |
| quality_level | string | Reporting and filtering |
| usable_for_caption | boolean | Mandatory gate |

No ASR model is loaded by this pipeline.

## 6. Phase 0 - Preflight and timeline

Responsibilities:

1. Resolve environment variables and paths.
2. Validate video IDs across frame, shot, and audio inputs.
3. Resolve all images before loading a GPU model.
4. Reject duplicate canonical frame IDs.
5. Sort frames and ASR segments by timestamp.
6. Validate shot membership and time ranges.
7. Compute immutable signatures for resume.

A frame result signature includes:

~~~text
image content hash
canonical_frame_id
model ID and revision
prompt version and hash
generation settings
schema version
~~~

A shot signature additionally includes ordered frame-caption signatures, ASR
feature IDs, shot bounds, and memory_before.

## 7. Phase 1 - Shot-aware frame selection

### 7.1 Goal

Reduce VLM calls while guaranteeing at least one directly observed frame per shot.

### 7.2 Selection roles

| Role | Meaning |
|---|---|
| anchor | Primary frame representing the shot |
| novel | Additional frame containing meaningful visual change |
| propagated | Redundant frame assigned to an anchor or novel source |
| forced | Selected by a safety rule despite low novelty |

### 7.3 Anchor rule

For the first implementation:

1. Collect frames assigned to the shot.
2. Exclude unreadable, blank, or known transition images when detectable.
3. Choose the valid frame closest to the temporal midpoint.
4. Fall back to the first valid frame.
5. Guarantee one anchor before novelty selection.

The midpoint is preferred because the first frame may still contain a transition
from the previous shot.

### 7.4 Novelty rule

Compare remaining frames with the most recent selected frame in the same shot.
The similarity implementation is pluggable:

~~~text
phash
existing_visual_embedding
hybrid
~~~

Use phash for the first implementation because it is CPU-friendly and requires
no additional GPU model. Thresholds are configurable and calibrated on L22_V012.
The 70 percent threshold in MMRS-LMF is a starting hypothesis, not a fixed value.

Safety constraints:

- Minimum selected frames per shot: 1.
- Maximum selected frames per shot: 3.
- Long or dynamic shots may force first/middle/last coverage.
- A frame without a valid propagation source must be selected for VLM inference.

### 7.5 Selection output

File:

~~~text
selection/frame_selection.jsonl
~~~

Schema: caption_frame_selection_v1.

~~~json
{
  "schema_version": "caption_frame_selection_v1",
  "run_id": "caption_l22_v012_YYYYMMDD_HHMMSS",
  "video_id": "L22_V012",
  "canonical_frame_id": "L22_V012_216",
  "keyframe_idx": 216,
  "timestamp_sec": 904.5,
  "shot_id": "L22_V012_shot_0078",
  "selection_role": "propagated",
  "selected_for_vlm": false,
  "selection_reasons": ["similar_to_anchor"],
  "source_frame_id": "L22_V012_215",
  "similarity_method": "phash",
  "similarity_score": 0.94,
  "novelty_score": 0.06,
  "warnings": []
}
~~~

## 8. Phase 2 - Visual frame captioning

### 8.1 Input

Normal frame captioning receives only:

~~~text
one full keyframe image
one visual-caption prompt
~~~

It does not receive ASR, OCR, objects, neighboring captions, or memory.

### 8.2 Baseline profile

~~~yaml
model_id: Qwen/Qwen2.5-VL-7B-Instruct
dtype: bfloat16
attention: flash_attention_2
batch_size: 2
max_visual_tokens: 1280
max_new_tokens: 64
do_sample: false
~~~

L22_V012 images are 1280 x 720, approximately 1176 Qwen visual tokens before
other processor adjustments. The reference run keeps a 1280-token cap. A later
experiment compares 768 tokens.

### 8.3 Prompt contract

~~~text
Inspect the scene, people, visible actions, spatial layout, and prominent
readable text. Resolve obvious ambiguities internally. Describe only what is
visually supported in the target frame. Do not infer identities, causes,
locations, or events. Return one concise English caption only.
~~~

The VLM returns plain text. The runtime wraps it in the canonical JSON record.
Requiring JSON from the VLM adds formatting risk without adding information.

### 8.4 Validation

- Non-empty after normalization.
- Maximum 320 characters.
- Recommended 5 to 60 words.
- No repeated sentence or phrase loop.
- No prompt, model, confidence, bounding-box, or metadata language.
- No phrases such as "the speaker says" because audio was not provided.
- One deterministic retry is allowed with a shorter correction prompt.

## 9. Phase 3 - Canonical frame captions

### 9.1 Invariant

Every valid keyframe produces exactly one canonical frame-caption row.

### 9.2 Modes

| Mode | Meaning |
|---|---|
| vlm_generated | Generated from this exact image |
| propagated | Copied from a selected frame in the same shot |
| fallback | Deterministic emergency output after failure |

### 9.3 Canonical file and schema

File:

~~~text
captions/frame_captions.jsonl
~~~

Schema: caption_frame_v2.

~~~json
{
  "schema_version": "caption_frame_v2",
  "run_id": "caption_l22_v012_YYYYMMDD_HHMMSS",
  "pipeline_version": "caption_audio_context_v1",
  "video_id": "L22_V012",
  "canonical_frame_id": "L22_V012_216",
  "keyframe_idx": 216,
  "source_frame_idx": 27000,
  "timestamp_sec": 904.5,
  "shot_id": "L22_V012_shot_0078",
  "image_relpath": "L22_V012/216.jpg",
  "selection": {
    "role": "anchor",
    "source_frame_id": null,
    "similarity_method": "phash",
    "similarity_score": null,
    "novelty_score": 1.0
  },
  "caption": {
    "text": "Several helmeted people stand beside a truck on an urban road.",
    "language": "en",
    "visual_only": true,
    "mode": "vlm_generated"
  },
  "generation": {
    "provider": "transformers",
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "model_revision": "",
    "prompt_version": "visual_frame_caption_v2",
    "prompt_hash": "sha256:...",
    "attempts": 1,
    "max_visual_tokens": 1280,
    "max_new_tokens": 64,
    "input_tokens": null,
    "output_tokens": null,
    "elapsed_ms": 1971,
    "batch_id": "vlm_batch_000052"
  },
  "quality": {
    "valid": true,
    "quality_level": "high",
    "word_count": 11,
    "char_count": 71,
    "warnings": []
  },
  "provenance": {
    "image_sha256": "...",
    "input_signature": "sha256:..."
  },
  "created_at": "2026-07-12T00:00:00Z"
}
~~~

For propagated captions, generation is null and selection.source_frame_id is
required. The complete row still contains identity, time, quality, and provenance.

## 10. Phase 4 - ASR alignment adapter

### 10.1 Overlap rule

An ASR segment overlaps a shot when:

~~~text
segment.end_sec >= shot.start_sec - padding_before_sec
and
segment.start_sec <= shot.end_sec + padding_after_sec
~~~

Initial values:

~~~yaml
padding_before_sec: 1.5
padding_after_sec: 1.5
max_audio_context_chars: 1200
~~~

### 10.2 Context construction

1. Keep only usable_for_caption rows.
2. Sort by start, end, and feature ID.
3. Deduplicate normalized repeated transcripts.
4. Preserve Vietnamese diacritics, names, and numbers.
5. Truncate after complete segments when possible.
6. Retain all feature IDs even when displayed text is truncated.
7. Do not summarize ASR with another model before ReCap.

### 10.3 Audio-context artifact

File:

~~~text
evidence/shot_audio_context.jsonl
~~~

Schema: caption_shot_audio_context_v1.

~~~json
{
  "schema_version": "caption_shot_audio_context_v1",
  "video_id": "L22_V012",
  "shot_id": "L22_V012_shot_0078",
  "shot_start_sec": 902.0,
  "shot_end_sec": 910.0,
  "alignment_window": {
    "start_sec": 900.5,
    "end_sec": 911.5
  },
  "feature_ids": ["L22_V012_asr_0042"],
  "text": "...",
  "language": "vi",
  "quality_levels": ["high"],
  "has_usable_audio": true,
  "text_truncated": false,
  "warnings": []
}
~~~

## 11. Phase 5 - Shot ReCap

### 11.1 Input

One shot request receives:

~~~text
shot ID and time range
ordered selected visual captions
caption modes and propagation statistics
aligned ASR text and feature IDs
compact memory_before
~~~

Normal text ReCap does not receive image paths, OCR, objects, or RAM tags.

The output field objects_involved is a semantic event field inferred only from
visual captions and aligned ASR. It is not an object-detector feature. Likewise,
scene is generated semantic text and is not sourced from RAM/scene tags.

### 11.2 Evidence rules

- Visual captions are observations.
- ASR is spoken context.
- Connect the two only when directly supported.
- If speech discusses something not visible, use wording such as "The narration
  discusses..." rather than claiming it is shown.
- Names from ASR identify a topic or person only when visual evidence does not
  contradict that assignment.

### 11.3 Compact memory

Memory is a bounded object:

~~~json
{
  "active_entities": [],
  "setting": "",
  "ongoing_topic": "",
  "ongoing_action": "",
  "last_event": ""
}
~~~

Rules:

- Maximum serialized size: 600 characters in the first test.
- Remove stale details instead of accumulating transcript text.
- Reset at a hard scene boundary.
- Memory is context, not proof that an action continues.
- ReCap is sequential within one video.
- At scale, batch the next ready shot across different videos.

### 11.4 Model profiles

Reference profile:

~~~yaml
model_id: Qwen/Qwen2.5-7B-Instruct
dtype: bfloat16
attention: flash_attention_2
max_new_tokens: 256
do_sample: false
~~~

Balanced profile after the reference run:

~~~yaml
model_id: Qwen/Qwen2.5-3B-Instruct
dtype: bfloat16
attention: flash_attention_2
max_new_tokens: 256
do_sample: false
~~~

Both profiles write exactly the same canonical schema.

### 11.5 Canonical shot output

File:

~~~text
captions/shot_captions.jsonl
~~~

Schema: caption_shot_v2.

~~~json
{
  "schema_version": "caption_shot_v2",
  "run_id": "caption_l22_v012_YYYYMMDD_HHMMSS",
  "pipeline_version": "caption_audio_context_v1",
  "video_id": "L22_V012",
  "shot_id": "L22_V012_shot_0078",
  "start_sec": 902.0,
  "end_sec": 910.0,
  "frame_ids": [
    "L22_V012_214",
    "L22_V012_215",
    "L22_V012_216"
  ],
  "selected_frame_ids": [
    "L22_V012_215",
    "L22_V012_216"
  ],
  "visual_evidence": [
    {
      "frame_id": "L22_V012_215",
      "timestamp_sec": 904.0,
      "selection_role": "anchor",
      "caption_text": "A truck is stopped on an urban road beside several people.",
      "caption_mode": "vlm_generated"
    }
  ],
  "audio_context": {
    "feature_ids": ["L22_V012_asr_0042"],
    "text": "...",
    "language": "vi",
    "has_usable_audio": true,
    "text_truncated": false
  },
  "memory_before": {
    "active_entities": [],
    "setting": "urban road",
    "ongoing_topic": "traffic incident",
    "ongoing_action": "",
    "last_event": "A report introduced the road incident."
  },
  "outputs": {
    "shot_caption": "Several people stand near a stopped truck on an urban road while the narration discusses the incident.",
    "temporal_caption": "The report continues with views of people gathered beside the truck.",
    "event_caption": "People gather near a stopped truck as a traffic incident is reported.",
    "action_state": "middle",
    "temporal_role": "continuation",
    "actors": ["several people"],
    "actions": ["standing", "gathering"],
    "objects_involved": ["truck"],
    "scene": "urban road",
    "trake_text": "Several people gather beside a stopped truck on an urban road during a report about a traffic incident."
  },
  "memory_after": {
    "active_entities": ["people near the truck"],
    "setting": "urban road",
    "ongoing_topic": "traffic incident",
    "ongoing_action": "people remain gathered near the truck",
    "last_event": "The report shows people gathered beside the stopped truck."
  },
  "generation": {
    "mode": "text_recap",
    "provider": "transformers",
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "model_revision": "",
    "prompt_version": "shot_recap_audio_context_v1",
    "prompt_hash": "sha256:...",
    "attempts": 1,
    "input_tokens": null,
    "output_tokens": null,
    "elapsed_ms": 0
  },
  "escalation": {
    "eligible": false,
    "triggered": false,
    "risk_score": 0.0,
    "reasons": [],
    "refined": false
  },
  "quality": {
    "valid": true,
    "visual_caption_coverage": 1.0,
    "direct_vlm_frame_ratio": 0.67,
    "has_usable_audio": true,
    "warnings": []
  },
  "provenance": {
    "frame_caption_signatures": ["sha256:..."],
    "audio_feature_ids": ["L22_V012_asr_0042"],
    "input_signature": "sha256:..."
  },
  "created_at": "2026-07-12T00:00:00Z"
}
~~~

### 11.6 Enums

~~~text
action_state:
  start | middle | end | transition | result | unknown

temporal_role:
  beginning | continuation | change | completion | unknown

generation.mode:
  text_recap | vlm_escalated | fallback
~~~

Known aliases should be normalized instead of causing a full regeneration. For
example, temporal_role=transition maps to change when the evidence supports it.

## 12. Phase 6 - Optional LVLM escalation

This phase is not required for the first coding milestone.

### 12.1 Candidate signals

- Missing or invalid generated frame caption.
- Low-confidence propagation in a dynamic shot.
- Strong disagreement between selected frame captions.
- Multiple distinct visual states in one shot.
- Failed text JSON after the repair attempt.
- Empty action fields for a visually dynamic shot.

### 12.2 Budget

~~~yaml
target_escalation_rate: 0.10
min_escalation_rate: 0.05
max_escalation_rate: 0.15
max_images_per_shot: 3
~~~

The percentage is a compute budget, not a required minimum. A clean video may
legitimately require fewer escalations.

### 12.3 Escalation input and output

Input:

~~~text
anchor image
highest-novelty image when present
last image when temporally useful
aligned ASR
baseline shot output
memory_before
~~~

The LVLM returns the same outputs and memory_after contract as text ReCap.
The baseline remains in an audit artifact.

### 12.4 One-GPU constraint

Escalation may require loading the VLM again after text ReCap. Therefore:

1. Implement and benchmark the pipeline without escalation first.
2. Build the complete escalation queue before loading the VLM again.
3. Process all candidates in one stage, not one model swap per shot.
4. If refined memory changes materially, replay only the bounded block up to the
   next hard scene reset.

## 13. Runtime artifacts

### 13.1 Memory ledger

File:

~~~text
state/recap_memory.jsonl
~~~

Schema: caption_recap_memory_v1.

Fields:

~~~text
video_id
shot_id
memory_before
memory_after
reset_applied
reset_reason
source_generation_mode
input_signature
~~~

### 13.2 Escalation audit

Files:

~~~text
runtime/escalation_candidates.jsonl
runtime/escalation_results.jsonl
~~~

Schema: caption_escalation_v1.

The result stores risk signals, baseline output, refined output, model metadata,
latency, and whether refinement replaced the canonical row.

## 14. Output layout

~~~text
caption_v2/L22_V012/
├── manifest/
│   ├── run_manifest.json
│   └── input_manifest.json
├── selection/
│   └── frame_selection.jsonl
├── evidence/
│   └── shot_audio_context.jsonl
├── captions/
│   ├── frame_captions.jsonl
│   └── shot_captions.jsonl
├── state/
│   └── recap_memory.jsonl
├── runtime/
│   ├── escalation_candidates.jsonl
│   └── escalation_results.jsonl
├── checkpoints/
│   ├── frame_vlm.jsonl
│   └── shot_recap.jsonl
└── reports/
    ├── caption_pipeline_report.json
    └── caption_pipeline_report.md
~~~

This design produces no retrieval index.

## 15. Run manifest

File: manifest/run_manifest.json.

Schema: caption_run_manifest_v1.

Required sections:

~~~json
{
  "schema_version": "caption_run_manifest_v1",
  "run_id": "caption_l22_v012_YYYYMMDD_HHMMSS",
  "pipeline_version": "caption_audio_context_v1",
  "video_id": "L22_V012",
  "created_at": "2026-07-12T00:00:00Z",
  "status": "running",
  "inputs": {
    "keyframes_root": "...",
    "keyframe_map": "...",
    "audio_features": "..."
  },
  "models": {
    "frame_vlm": {},
    "shot_text_llm": {},
    "escalation_vlm": {}
  },
  "prompts": {},
  "config_hash": "sha256:...",
  "environment": {
    "cuda_visible_devices": "1",
    "torch_version": "",
    "transformers_version": "",
    "flash_attn_version": ""
  },
  "counts": {},
  "timings": {},
  "warnings": []
}
~~~

## 16. Checkpoint and resume

Frame key:

~~~text
canonical_frame_id + input_signature
~~~

Shot key:

~~~text
shot_id + input_signature
~~~

Rules:

- Reuse valid non-fallback rows with matching signatures.
- A configuration may explicitly allow fallback reuse.
- Model, prompt, evidence, schema, or memory changes invalidate the row.
- Write checkpoints atomically at configurable intervals.
- Sort final JSONL deterministically by timestamp and ID.
- Never use output row order as a join key.

## 17. Failure policy

| Failure | Required behavior |
|---|---|
| Missing image | Detect during preflight |
| VLM OOM | Reduce batch size and resume; do not silently change model |
| Invalid VLM caption | One correction retry, then fallback/propagation warning |
| Invalid text JSON | Normalize aliases, then one repair retry |
| Text generation failure | Deterministic marked fallback |
| Missing audio | Continue visual-only with has_usable_audio=false |
| Memory overflow | Truncate fields by priority and warn |
| Interrupted run | Resume by matching signatures |

Strict mode turns acceptance failures into a non-zero exit code. Record-level
failures should not destroy a long batch.

## 18. Report metrics

### Input and coverage

~~~text
num_frames
num_shots
num_audio_features_total
num_audio_features_usable
num_shots_with_audio
audio_context_coverage
~~~

### Selection and savings

~~~text
num_anchor_frames
num_novel_frames
num_forced_frames
num_propagated_frames
num_vlm_calls
vlm_selection_ratio
caption_reuse_ratio
estimated_vlm_calls_avoided
~~~

### Generation quality

~~~text
num_frame_captions_valid
num_frame_captions_fallback
num_shot_captions_valid
num_shot_captions_fallback
num_empty_captions
num_duplicate_ids
text_retry_rate
~~~

### Runtime

~~~text
model_load_seconds
frame_vlm_seconds
propagation_seconds
audio_alignment_seconds
shot_recap_seconds
escalation_seconds
total_seconds
peak_gpu_memory_mb
input_tokens_total
output_tokens_total
~~~

### Escalation

~~~text
num_escalation_candidates
num_escalated
escalation_rate
escalation_reason_counts
num_refined_records_accepted
~~~

## 19. L22_V012 acceptance checks

### 19.1 Blocking

1. Produce exactly 283 unique frame rows for the known input.
2. Produce exactly 102 unique shot rows under the current baseline grouping.
3. Frame and shot timelines are monotonic.
4. Every frame belongs to exactly one shot.
5. Every shot has at least one direct VLM frame caption.
6. Every propagated frame references a selected source in the same shot.
7. Every frame caption is non-empty after fallback handling.
8. Every shot has non-empty shot_caption, event_caption, and trake_text.
9. All enums satisfy their contracts.
10. No raw OCR or object-detection evidence field appears in this branch.
11. No duplicate frame ID, shot ID, or checkpoint key.

### 19.2 Initial warnings

- VLM selection ratio above 0.60: insufficient reuse.
- VLM selection ratio below 0.30: possible over-propagation.
- Fallback rate above 0.05.
- Text retry rate above 0.05.
- Unexpected drop in audio context coverage.
- Propagated row below the configured similarity gate.
- Escalation rate above 0.15 when enabled.

These warning thresholds require calibration after the first run.

## 20. L22_V012 experiment sequence

### Experiment A - Reference schema run

~~~text
Frame VLM: Qwen2.5-VL-7B
Selection: one anchor per shot
Shot LLM: Qwen2.5-7B
Escalation: disabled
Visual token cap: 1280
~~~

Purpose: validate schemas, prompts, propagation, ReCap memory, and acceptance.

### Experiment B - Novelty run

~~~text
Frame VLM: Qwen2.5-VL-7B
Selection: anchor + novelty, maximum 3 per shot
Shot LLM: Qwen2.5-7B
Escalation: disabled
~~~

Purpose: compare visual/action coverage against one-anchor selection.

### Experiment C - Balanced text model

~~~text
Reuse Experiment B frame captions
Shot LLM: Qwen2.5-3B
~~~

Purpose: measure speed, JSON validity, coherence, and TRAKE quality without
rerunning the VLM.

### Experiment D - Lower visual token budget

~~~text
Reuse Experiment B selection
Visual token cap: 768
~~~

Purpose: measure visual quality and latency independently from model-size changes.

### Experiment E - Escalation

Enable the 5-15 percent risk budget only after A-D produce stable outputs.

## 21. Suggested implementation boundaries

These are proposed modules, not files created by this design:

~~~text
caption_pipeline_v2/
├── config.py
├── schemas.py
├── input_contracts.py
├── timeline.py
├── shot_builder.py
├── frame_selector.py
├── vlm_frame_captioner.py
├── frame_propagator.py
├── audio_context_adapter.py
├── recap_memory.py
├── shot_recap.py
├── escalation_router.py
├── validators.py
├── checkpoint.py
├── report.py
└── runner.py
~~~

Ownership:

- frame_selector never loads a generative model.
- vlm_frame_captioner never loads ASR or text-fusion evidence.
- audio_context_adapter never modifies audio output.
- shot_recap never reads OCR or object files.
- schemas contains versions and enum contracts.
- runner manages model residency and stage order, not caption semantics.

## 22. GPU execution

### One GPU

~~~text
CPU preflight and selection
-> load VLM once
-> caption selected frames in batches
-> unload VLM
-> CPU propagation and audio alignment
-> load text LLM once
-> process shots sequentially with resume
-> unload text LLM
-> optional batched escalation stage
-> report
~~~

### Two GPUs

~~~text
GPU 0: persistent frame/escalation VLM
GPU 1: text ReCap model
CPU: selection, propagation, ASR alignment, validation, I/O
~~~

For one test video, two GPUs do not remove ReCap's within-video dependency.

## 23. Contract for later 1M-frame runs

The implementation must not assume a single video:

1. Partition by video_id.
2. Do not split one ReCap sequence without an explicit memory boundary.
3. Batch VLM requests with similar image token budgets.
4. Batch text requests across videos, one ready shot from each video.
5. Keep models resident for large queues; never swap per video.
6. Persist incremental checkpoints.
7. Use signatures to avoid recomputing unchanged inputs.
8. Preserve generated-versus-propagated provenance.
9. Keep all selection thresholds configurable by profile.

## 24. Research alignment

The design adapts, rather than copies, the papers:

- U-CESE keyframe context and internal visual reasoning become optional reasoning
  for selected frames and future hard-shot escalation.
- U-CESE ReCap becomes compact explicit memory between shot requests.
- MMRS-LMF caption reuse becomes auditable anchor/novelty propagation.
- FuseCap-style OCR/object injection is excluded because this project observed
  caption noise from those features.

References:

- U-CESE: https://arxiv.org/html/2605.23274v1
- MMRS-LMF: https://easychair.org/publications/preprint/dfqX

## 25. First coding milestone definition of done

The first implementation is done when:

1. It runs end to end on L22_V012 with existing keyframes and audio features.
2. It writes all canonical artifacts except optional escalation results.
3. It produces 283 frame rows and 102 shot rows for the known baseline.
4. It performs no frame-level text LLM fusion.
5. It performs one normal text generation per shot.
6. It never reads OCR or object-detection output.
7. It resumes safely without regenerating matching rows.
8. It reports calls, reuse savings, tokens when available, latency, retries,
   fallbacks, and audio coverage.
9. Strict acceptance checks pass.
10. Reference and balanced text-model profiles reuse the same frame captions.
