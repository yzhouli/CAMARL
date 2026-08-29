# CAMARL

Official implementation of **CAMARL: Cost-Aware Multi-Agent Reinforcement Learning for Information Diffusion Reranking**.

CAMARL uses a coordinator to decide whether to call semantic, profile, and
topology experts or to stop and return a ranking. This repository contains the
source code only. It does **not** contain MosaicDiff data, generated processed
files, the base language model, or CAMARL weights.

## Repository contents

```text
CAMARL/
├── README.md
├── model/
│   ├── configs/paper.env.example
│   ├── requirements.txt
│   └── scripts/                  CAMARL training and evaluation code
├── MosaicDiff/
│   ├── raw/                      empty; place downloaded data here
│   ├── processed/                empty; preprocessing writes outputs here
│   └── processing/               dataset preprocessing code
```

`MosaicDiff/processed/`, `outputs/`, and `checkpoints/` are generated locally.
Do not add their generated contents or model weights to Git.

## 1. Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/yzhouli/CAMARL.git
cd CAMARL

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r model/requirements.txt
```

Install [vLLM](https://docs.vllm.ai/) separately with a version compatible
with the installed CUDA and PyTorch versions. The examples below use vLLM's
OpenAI-compatible local server.

Set the working paths. `BASE_MODEL` must point to a locally downloaded base
model; no model weights are included in this repository.

```bash
export CAMARL_ROOT="$PWD"
export MOSAICDIFF_ROOT="$CAMARL_ROOT/MosaicDiff"
export RAW_DIR="$MOSAICDIFF_ROOT/raw"
export PROCESSED_DIR="$MOSAICDIFF_ROOT/processed"
export PROTOCOL_DIR="$PROCESSED_DIR/protocol"
export GRAPHHARD_DIR="$PROCESSED_DIR/graphhard_large"
export MEDIA_DIR="$PROCESSED_DIR/topic_frames"
export TRAINING_DIR="$PROCESSED_DIR/camarl_training"
export BASE_MODEL=/absolute/path/to/Qwen3.5-4B
```

## 2. Download MosaicDiff

MosaicDiff is distributed separately because it is too large for GitHub.

```text
Dataset download URL: DATASET_DOWNLOAD_URL_TO_BE_ADDED
```

Download and extract the dataset, then copy the **contents** of the extracted
directory into `MosaicDiff/raw/`:

```bash
export DOWNLOADED_DATASET=/absolute/path/to/extracted/MosaicDiff
mkdir -p "$RAW_DIR"
rsync -a "$DOWNLOADED_DATASET/" "$RAW_DIR/"
```

The resulting layout must contain at least:

```text
MosaicDiff/raw/
├── cascades.txt
├── edges.txt
├── news_all.pkl
├── users_all.pkl
├── test_aligned.pkl
└── mm/mm/                       images and videos
```

Check the required inputs before preprocessing:

```bash
for item in cascades.txt edges.txt news_all.pkl users_all.pkl test_aligned.pkl mm/mm
do
  test -e "$RAW_DIR/$item" || { echo "Missing: $RAW_DIR/$item"; exit 1; }
done
```

Only load the pickle files from the official, trusted dataset download.

## 3. Preprocess MosaicDiff

Run all commands from the repository root with the environment activated.
Generated files are written to `MosaicDiff/processed/`.

### 3.1 Build the fixed data protocol

This creates cascade-level train/validation/test manifests and the base
candidate-pool records.

```bash
python MosaicDiff/processing/build_protocol_pools.py \
  --data-dir "$RAW_DIR" \
  --output-dir "$PROTOCOL_DIR" \
  --device cuda:0
```

### 3.2 Train the five-seed graph miner

```bash
python MosaicDiff/processing/train_eval_twotower_base.py \
  --protocol-dir "$PROTOCOL_DIR" \
  --users-path "$RAW_DIR/users_all.pkl" \
  --device cuda:0
```

This writes the five `interaction_twotower_seed*.pt` miner checkpoints into
`$PROTOCOL_DIR`. They are preprocessing artifacts, not CAMARL policy weights.

### 3.3 Build graph-hard candidate pools

The following command generates the N=1000, 1500, and 2000 validation and test
pools used by the large-pool CAMARL experiments:

```bash
python MosaicDiff/processing/build_graphhard_pools.py \
  --protocol-dir "$PROTOCOL_DIR" \
  --data-dir "$RAW_DIR" \
  --output-dir "$GRAPHHARD_DIR" \
  --negative-universe-size 1999 \
  --pool-sizes 1000 1500 2000 \
  --device cuda:0
```

Record the hash of the newly generated frozen protocol report. Later commands
use it to prevent accidental mixing of different candidate pools.

```bash
export PROTOCOL_SHA256="$(python -c 'import hashlib,os,pathlib; p=pathlib.Path(os.environ["GRAPHHARD_DIR"])/"graphhard_protocol_report.json"; print(hashlib.sha256(p.read_bytes()).hexdigest())')"
echo "$PROTOCOL_SHA256"
```

### 3.4 Prepare visual inputs

Static images are resized and videos are converted to four-frame contact
sheets. Missing media are recorded and use CAMARL's text-only fallback.

```bash
python MosaicDiff/processing/prepare_topic_media.py \
  --news "$RAW_DIR/news_all.pkl" \
  --source-dir "$RAW_DIR/mm/mm" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --output-dir "$MEDIA_DIR" \
  --splits validation test \
  --allow-missing
```

## 4. Build CAMARL training data

CAMARL training first caches the outputs of all three experts on the
**validation split only**. Start the base-model server in one terminal:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$BASE_MODEL" \
  --served-model-name Qwen3.5_4B \
  --port 8300 \
  --max-model-len 32768 \
  --dtype bfloat16
```

In another terminal, restore the environment variables from Sections 1 and 3,
then build the strict fixed-all cache:

```bash
python model/scripts/eval_no_grpo_graphhard.py \
  --method fixed_all \
  --split validation \
  --served-model-name Qwen3.5_4B \
  --checkpoint "$BASE_MODEL" \
  --ports 8300 \
  --server-max-model-len 32768 \
  --pool-sizes 1000 1500 2000 \
  --data-dir "$RAW_DIR" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --media-dir "$MEDIA_DIR" \
  --output-dir "$PROCESSED_DIR/validation_cache" \
  --fail-on-request-error
```

Convert the cache into validation-only coordinator states:

```bash
mkdir -p "$TRAINING_DIR"
export CAMARL_DATASET="$TRAINING_DIR/coordinator_validation_states.jsonl"

python model/scripts/build_magrpo_coordinator_dataset.py \
  --validation-report "$PROCESSED_DIR/validation_cache/fixed_all_results.json" \
  --output "$CAMARL_DATASET" \
  --pool-sizes 1000 1500 2000 \
  --data-dir "$RAW_DIR" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256"
```

The command also creates
`coordinator_validation_states.jsonl.manifest.json`. The training script checks
that this manifest is validation-only and does not expose target information.

## 5. Train CAMARL

The example below trains the independent N=1000 coordinator with the paper
configuration: group size G=4, LoRA rank 16, alpha 32, dropout 0.05, learning
rate 5e-7, and 200 optimization steps.

```bash
torchrun --standalone --nproc_per_node=1 \
  model/scripts/train_magrpo_coordinator.py \
  --dataset "$CAMARL_DATASET" \
  --model "$BASE_MODEL" \
  --output-dir "$CAMARL_ROOT/outputs/train_N1000" \
  --pool-size 1000 \
  --num-generations 4 \
  --max-steps 200 \
  --learning-rate 5e-7 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

Checkpoints are saved every 50 steps by default. The final LoRA adapter is
written to `outputs/train_N1000/final_adapter/`. Train separate policies for
other pool sizes by changing `--pool-size` and the output directory.

## 6. Validate and freeze a policy

Before test inference, serve a candidate LoRA adapter and audit it on the
validation-dev subset. This example audits the final adapter:

```bash
export CANDIDATE_ADAPTER="$CAMARL_ROOT/outputs/train_N1000/final_adapter"

CUDA_VISIBLE_DEVICES=0 vllm serve "$BASE_MODEL" \
  --served-model-name Qwen3.5_4B \
  --enable-lora \
  --lora-modules CAMARL_CANDIDATE="$CANDIDATE_ADAPTER" \
  --port 8400 \
  --max-model-len 32768 \
  --dtype bfloat16
```

Run the audit from another terminal:

```bash
mkdir -p "$CAMARL_ROOT/outputs/audits"

python model/scripts/audit_magrpo_coordinator_policy.py \
  --dataset "$CAMARL_DATASET" \
  --coordinator-model-name CAMARL_CANDIDATE \
  --adapter-path "$CANDIDATE_ADAPTER" \
  --pool-size 1000 \
  --ports 8400 \
  --server-max-model-len 32768 \
  --fail-on-request-error \
  --output "$CAMARL_ROOT/outputs/audits/final_adapter.json"
```

For checkpoint selection, repeat the audit for `checkpoint-50`,
`checkpoint-100`, `checkpoint-150`, and `checkpoint-200`, then pass all audit
JSON files to `--audits`. The minimal single-candidate freeze command is:

```bash
python model/scripts/select_freeze_magrpo_policy.py \
  --audits "$CAMARL_ROOT/outputs/audits/final_adapter.json" \
  --pool-size 1000 \
  --frozen-dir "$CAMARL_ROOT/checkpoints/camarl_N1000" \
  --output "$CAMARL_ROOT/checkpoints/camarl_N1000_policy.json"
```

The frozen directory must not already contain files.

## 7. Inference and evaluation

Stop the candidate server and serve the frozen adapter:

```bash
export CAMARL_ADAPTER="$CAMARL_ROOT/checkpoints/camarl_N1000"

CUDA_VISIBLE_DEVICES=0 vllm serve "$BASE_MODEL" \
  --served-model-name Qwen3.5_4B \
  --enable-lora \
  --lora-modules CAMARL_N1000="$CAMARL_ADAPTER" \
  --port 8400 \
  --max-model-len 32768 \
  --dtype bfloat16
```

Run test inference for the independently trained N=1000 policy:

```bash
python model/scripts/eval_magrpo_coordinator_graphhard.py \
  --coordinator-model-name CAMARL_N1000 \
  --base-model-name Qwen3.5_4B \
  --coordinator-adapter "$CAMARL_ADAPTER" \
  --policy-manifest "$CAMARL_ROOT/checkpoints/camarl_N1000_policy.json" \
  --trained-pool-size 1000 \
  --pool-sizes 1000 \
  --split test \
  --seeds 13 21 34 55 89 \
  --ports 8400 \
  --server-max-model-len 32768 \
  --checkpoint "$BASE_MODEL" \
  --data-dir "$RAW_DIR" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --media-dir "$MEDIA_DIR" \
  --output-dir "$CAMARL_ROOT/outputs/test_N1000" \
  --fail-on-request-error
```

Results are written to
`outputs/test_N1000/magrpo_coordinator_results.json`. A policy trained for one
pool size is intentionally evaluated only on that same pool size.

## Notes

- Do not commit raw or processed data, model weights, checkpoints, or outputs.
- The dataset download URL above is a placeholder and must be replaced after
  the external MosaicDiff release is available.
- MosaicDiff and model weights require their own licenses before redistribution.
