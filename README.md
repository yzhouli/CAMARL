# CAMARL

Official release scaffold for **CAMARL: Cost-Aware Multi-Agent Reinforcement
Learning for Information Diffusion Reranking**.

CAMARL uses a trainable coordinator to decide whether to call semantic,
profile, and topology experts or to stop and return a ranking. The coordinator
is optimized with group-relative reinforcement learning, a validation-only
cost-aware reference policy, and an explicit expert-acquisition penalty.

## Release contents

```text
CAMARL/
├── scripts/       data preparation, training, selection, and evaluation
├── configs/       paper configuration example
└── docs/          data contract and reproducibility notes
```

Generated outputs, datasets, and model weights are not included in this GitHub
repository. Model weights should be uploaded as a separate artifact only after
their license and model card are finalized. The base model is never included.

## Dataset

The CAMARL dataset is distributed separately because it is too large for this
source repository. Download it from the external artifact page below, then
extract it anywhere on your machine:

```text
REPLACE_WITH_CAMARL_DATASET_DOWNLOAD_URL
```

Replace this placeholder with the final dataset URL before publishing the
GitHub repository. The dataset artifact must carry its own license, privacy
review, dataset card, and SHA-256 manifest.

## Installation

Linux, Python 3.10 or 3.11, CUDA, and one or more recent NVIDIA GPUs are
recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install `vllm` separately using the version compatible with your CUDA and
PyTorch stack. The evaluation scripts talk to vLLM through its OpenAI-compatible
local endpoint; no paid API is required.

## Configure local paths

After downloading and extracting the separate dataset artifact, point
`CAMARL_DATASET_ROOT` to its local directory:

```bash
export CAMARL_ROOT="$PWD"
export CAMARL_DATASET_ROOT=/absolute/path/to/CAMARL-Dataset
export BASE_MODEL=/path/to/Qwen3.5-4B
export GRAPHHARD_DIR="$CAMARL_DATASET_ROOT/processed/graphhard_large"
export PROTOCOL_SHA256="$(sha256sum "$GRAPHHARD_DIR/graphhard_protocol_report.json" | cut -d ' ' -f1)"
export CAMARL_TRAINING_DATASET="$CAMARL_DATASET_ROOT/processed/camarl_training/coordinator_validation_states.jsonl"
```

All paths are explicit command-line arguments. No script contains a private
server path, key, or host address.

## Pipeline

### 1. Build the leakage-controlled protocol

This step creates deterministic train/validation/test manifests and the base
candidate pools from the raw dataset.

```bash
python scripts/build_protocol_pools.py \
  --data-dir "$CAMARL_DATASET_ROOT/raw" \
  --output-dir "$CAMARL_DATASET_ROOT/processed/protocol" \
  --device cuda:0

python scripts/train_eval_twotower_base.py \
  --protocol-dir "$CAMARL_DATASET_ROOT/processed/protocol" \
  --users-path "$CAMARL_DATASET_ROOT/raw/users_all.pkl" \
  --device cuda:0

python scripts/build_graphhard_pools.py \
  --protocol-dir "$CAMARL_DATASET_ROOT/processed/protocol" \
  --data-dir "$CAMARL_DATASET_ROOT/raw" \
  --output-dir "$CAMARL_DATASET_ROOT/processed/graphhard_large" \
  --negative-universe-size 1999 \
  --pool-sizes 1000 1500 2000 \
  --device cuda:0
```

The released processed protocol can be used directly. Rebuilding is useful for
auditing and should reproduce the protocol report for the same raw snapshot.

### 2. Prepare topic media

```bash
python scripts/prepare_topic_media.py \
  --news "$CAMARL_DATASET_ROOT/raw/news_all.pkl" \
  --source-dir "$CAMARL_DATASET_ROOT/raw/mm/mm" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --output-dir "$CAMARL_DATASET_ROOT/processed/topic_frames" \
  --splits validation test
```

### 3. Start local base-model servers

Start one or more vLLM OpenAI-compatible servers. Example:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$BASE_MODEL" \
  --served-model-name Qwen3.5_4B \
  --port 8300 \
  --max-model-len 32768 \
  --dtype bfloat16
```

### 4. Cache validation expert outputs

The `fixed_all` validation run executes every expert once. Its cached expert
observations are used to construct the validation-only CAMARL training states.

```bash
python scripts/eval_no_grpo_graphhard.py \
  --method fixed_all \
  --split validation \
  --served-model-name Qwen3.5_4B \
  --checkpoint "$BASE_MODEL" \
  --ports 8300 \
  --data-dir "$CAMARL_DATASET_ROOT/raw" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --media-dir "$CAMARL_DATASET_ROOT/processed/topic_frames" \
  --output-dir outputs/validation_cache \
  --fail-on-request-error
```

### 5. Build validation-only coordinator states

```bash
python scripts/build_magrpo_coordinator_dataset.py \
  --validation-report outputs/validation_cache/fixed_all_results.json \
  --output outputs/coordinator_validation_states.jsonl \
  --data-dir "$CAMARL_DATASET_ROOT/raw" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256"
```

The generated manifest records that test records were not used and that the
hidden target is not visible to the model prompt.

The external dataset release already includes the audited files under
`processed/validation_cache/` and `processed/camarl_training/`. To reproduce
the released checkpoint directly, use those files and skip Steps 4--5; rerun
the steps only when auditing or rebuilding the training data.

### 6. Train CAMARL

The source default selects group size `G=4`, matching the final sensitivity
analysis. The reward mixture is
`0.65 ranking utility + 0.30 reference-action alignment + 0.05 valid format`,
with acquisition penalty `0.10` and invalid-format reward `-0.50`.

```bash
torchrun --standalone --nproc_per_node=1 \
  scripts/train_magrpo_coordinator.py \
  --dataset "$CAMARL_TRAINING_DATASET" \
  --model "$BASE_MODEL" \
  --output-dir outputs/train_N1000 \
  --pool-size 1000 \
  --num-generations 4 \
  --max-steps 200 \
  --learning-rate 5e-7 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

Batch size and gradient accumulation must make the accumulated global
completion batch divisible by `--num-generations`.

### 7. Select, freeze, and evaluate

Use `audit_magrpo_coordinator_policy.py` on checkpoints 50/100/150/200, then
select the best validation-dev checkpoint:

```bash
# Repeat for each served checkpoint/adapter name.
python scripts/audit_magrpo_coordinator_policy.py \
  --dataset "$CAMARL_TRAINING_DATASET" \
  --coordinator-model-name CAMARL_CKPT50 \
  --adapter-path outputs/train_N1000/checkpoint-50 \
  --pool-size 1000 \
  --ports 8400 \
  --fail-on-request-error \
  --output outputs/audits/checkpoint-50.json

python scripts/select_freeze_magrpo_policy.py \
  --audits outputs/audits/checkpoint-*.json \
  --pool-size 1000 \
  --frozen-dir checkpoints/camarl_N1000 \
  --output checkpoints/camarl_N1000_policy.json
```

Serve the base model with the selected LoRA adapter and run:

```bash
python scripts/eval_magrpo_coordinator_graphhard.py \
  --coordinator-model-name CAMARL_N1000 \
  --base-model-name Qwen3.5_4B \
  --coordinator-adapter checkpoints/camarl_N1000 \
  --policy-manifest checkpoints/camarl_N1000_policy.json \
  --trained-pool-size 1000 \
  --pool-sizes 1000 \
  --ports 8400 \
  --checkpoint "$BASE_MODEL" \
  --data-dir "$CAMARL_DATASET_ROOT/raw" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --media-dir "$CAMARL_DATASET_ROOT/processed/topic_frames" \
  --output-dir outputs/test_N1000 \
  --fail-on-request-error
```

## Reproducibility and safety

- Training states are built from validation only; test labels are rejected.
- Candidate order is deterministically shuffled by seed.
- The protocol hash is checked before evaluation.
- Output manifests bind datasets, adapters, configuration, and checkpoints by
  SHA-256.
- Pickle files must only be loaded from this trusted release snapshot.

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for details.

## Citation

See `CITATION.cff`. Update the publication venue, year, DOI, and repository URL
when the paper and artifacts are public.

## License

The source code is released under the MIT License. Dataset content, model
weights, and the base model are separate artifacts and require their own
license and privacy review before publication.
