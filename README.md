# CAMARL

Official implementation of **CAMARL: Cost-Aware Multi-Agent Reinforcement
Learning for Information Diffusion Reranking**.

CAMARL trains a coordinator to decide whether to call semantic, profile, and
topology experts, or to stop and return a ranking. The Git repository contains
the model and MosaicDiff preprocessing code. A local release bundle may also
contain the processed research snapshot under `MosaicDiff/processed/`; raw and
processed binary artifacts and model weights are not tracked by Git.

## Repository layout

```text
CAMARL/
├── README.md                 this complete guide
├── LICENSE                   MIT license for source code
├── CITATION.cff              citation metadata
├── model/
│   ├── scripts/              preprocessing, training, and evaluation
│   ├── configs/              paper configuration example
│   └── requirements.txt
└── MosaicDiff/
    ├── raw/                  empty in the repository; install separately
    ├── processed/            local processed snapshot or regenerated files
    └── processing/           standalone MosaicDiff preprocessing scripts
```

The separately uploaded raw artifact has this local staging directory:

```text
/path/to/downloaded/dataset/
├── cascades.txt
├── edges.txt
├── news_all.pkl
├── users_all.pkl
├── test_aligned.pkl
├── mm/mm/
└── additional frozen raw artifacts
```

## 1. Download and install raw MosaicDiff data

Raw dataset download page:

```text
REPLACE_WITH_RAW_DATASET_DOWNLOAD_URL
```

After downloading and extracting the external `dataset` artifact, copy its
contents into the empty `MosaicDiff/raw/` directory:

```bash
cd /path/to/CAMARL
export CAMARL_ROOT="$PWD"
export RAW_DOWNLOAD_DIR=/path/to/downloaded/dataset

mkdir -p "$CAMARL_ROOT/MosaicDiff/raw"
rsync -a "$RAW_DOWNLOAD_DIR/" "$CAMARL_ROOT/MosaicDiff/raw/"
```

The raw artifact may contain its integrity manifest in addition to the files
used by the code. Extra documentation or manifest files under `raw/` do not
affect the loaders.

Verify the required inputs before preprocessing:

```bash
for path in \
  cascades.txt edges.txt news_all.pkl users_all.pkl test_aligned.pkl mm/mm
do
  test -e "$CAMARL_ROOT/MosaicDiff/raw/$path" || {
    echo "missing MosaicDiff/raw/$path" >&2
    exit 1
  }
done
```

Never load pickle files from an untrusted or unverified download. Python
pickle can execute code during deserialization.

## 2. Environment

Linux, Python 3.10 or 3.11, CUDA, and recent NVIDIA GPUs are recommended.

```bash
cd /path/to/CAMARL
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r model/requirements.txt
```

Install `vllm` separately using the version compatible with the selected CUDA
and PyTorch stack. Evaluation uses vLLM's OpenAI-compatible local endpoint; no
paid API is required.

Configure paths:

```bash
export CAMARL_ROOT=/absolute/path/to/CAMARL
export CAMARL_MODEL_ROOT="$CAMARL_ROOT/model"
export MOSAICDIFF_ROOT="$CAMARL_ROOT/MosaicDiff"
export BASE_MODEL=/absolute/path/to/Qwen3.5-4B
export GRAPHHARD_DIR="$MOSAICDIFF_ROOT/processed/graphhard_large"
export CAMARL_TRAINING_DATASET="$MOSAICDIFF_ROOT/processed/camarl_training/coordinator_validation_states.jsonl"
export PROTOCOL_SHA256="$(python -c 'import hashlib,os,pathlib; p=pathlib.Path(os.environ["GRAPHHARD_DIR"])/"graphhard_protocol_report.json"; print(hashlib.sha256(p.read_bytes()).hexdigest())')"
```

The base model and LoRA adapters are not included in this repository.

## 3. Use the processed snapshot

If `MosaicDiff/processed/` was obtained with the release bundle, the following
artifacts can be used directly:

```text
processed/protocol/             deterministic split and base-pool artifacts
processed/graphhard_v2/         N=20/50/100/500 candidate pools
processed/graphhard_large/      N=1000/1500/2000 candidate pools
processed/topic_frames/         prepared image/contact-sheet inputs
processed/validation_cache/     fixed-all validation expert cache
processed/camarl_training/      validation-only CAMARL coordinator states
```

The released large-pool protocol report must have this SHA-256 value:

```text
c7360ba282746a0fa9c6dee8a995d8c172197a0ceeb64937240662738e460c6f
```

## 4. Rebuild MosaicDiff processed data

The commands below are run from the CAMARL repository root.

### 4.1 Build leakage-controlled splits and base pools

```bash
python MosaicDiff/processing/build_protocol_pools.py \
  --data-dir "$MOSAICDIFF_ROOT/raw" \
  --output-dir "$MOSAICDIFF_ROOT/processed/protocol" \
  --device cuda:0
```

### 4.2 Train the five-seed graph miner

```bash
python MosaicDiff/processing/train_eval_twotower_base.py \
  --protocol-dir "$MOSAICDIFF_ROOT/processed/protocol" \
  --users-path "$MOSAICDIFF_ROOT/raw/users_all.pkl" \
  --device cuda:0
```

### 4.3 Build N=1000/1500/2000 graph-hard pools

```bash
python MosaicDiff/processing/build_graphhard_pools.py \
  --protocol-dir "$MOSAICDIFF_ROOT/processed/protocol" \
  --data-dir "$MOSAICDIFF_ROOT/raw" \
  --output-dir "$MOSAICDIFF_ROOT/processed/graphhard_large" \
  --negative-universe-size 1999 \
  --pool-sizes 1000 1500 2000 \
  --device cuda:0
```

### 4.4 Prepare image and video contact sheets

```bash
python MosaicDiff/processing/prepare_topic_media.py \
  --news "$MOSAICDIFF_ROOT/raw/news_all.pkl" \
  --source-dir "$MOSAICDIFF_ROOT/raw/mm/mm" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --output-dir "$MOSAICDIFF_ROOT/processed/topic_frames" \
  --splits validation test \
  --allow-missing
```

`--allow-missing` records topics without a source medium in the generated
manifest while allowing the text-only fallback used by CAMARL.

## 5. Build CAMARL training states

The released processed snapshot already contains the fixed-all cache and
training states. Rebuild them only when auditing or regenerating the protocol.

First start one or more base-model servers, for example:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$BASE_MODEL" \
  --served-model-name Qwen3.5_4B \
  --port 8300 \
  --max-model-len 32768 \
  --dtype bfloat16
```

Cache all three expert outputs on validation data. The explicit pool sizes are
required because `graphhard_large` does not contain the smaller v2 pools.

```bash
python model/scripts/eval_no_grpo_graphhard.py \
  --method fixed_all \
  --split validation \
  --served-model-name Qwen3.5_4B \
  --checkpoint "$BASE_MODEL" \
  --ports 8300 \
  --pool-sizes 1000 1500 2000 \
  --data-dir "$MOSAICDIFF_ROOT/raw" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --media-dir "$MOSAICDIFF_ROOT/processed/topic_frames" \
  --output-dir "$MOSAICDIFF_ROOT/processed/validation_cache" \
  --fail-on-request-error
```

Build validation-only coordinator states:

```bash
python model/scripts/build_magrpo_coordinator_dataset.py \
  --validation-report "$MOSAICDIFF_ROOT/processed/validation_cache/fixed_all_results.json" \
  --output "$CAMARL_TRAINING_DATASET" \
  --pool-sizes 1000 1500 2000 \
  --data-dir "$MOSAICDIFF_ROOT/raw" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256"
```

The generated sidecar must state `split=validation`,
`test_records_used=false`, and `target_visible_to_model=false`.

## 6. Train CAMARL

The paper configuration uses group size `G=4`, LoRA rank 16, alpha 32,
dropout 0.05, learning rate `5e-7`, and 200 optimization steps. Its reward is
`0.65 ranking utility + 0.30 reference-action alignment + 0.05 valid format`,
with normalized expert-acquisition penalty `0.10` and invalid-format reward
`-0.50`.

```bash
torchrun --standalone --nproc_per_node=1 \
  model/scripts/train_magrpo_coordinator.py \
  --dataset "$CAMARL_TRAINING_DATASET" \
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

The accumulated global completion batch must be divisible by
`--num-generations`.

## 7. Select and freeze a validation checkpoint

Serve each candidate adapter through vLLM, audit checkpoints 50/100/150/200,
and select the best validation-dev checkpoint:

```bash
python model/scripts/audit_magrpo_coordinator_policy.py \
  --dataset "$CAMARL_TRAINING_DATASET" \
  --coordinator-model-name CAMARL_CKPT50 \
  --adapter-path "$CAMARL_ROOT/outputs/train_N1000/checkpoint-50" \
  --pool-size 1000 \
  --ports 8400 \
  --fail-on-request-error \
  --output "$CAMARL_ROOT/outputs/audits/checkpoint-50.json"

python model/scripts/select_freeze_magrpo_policy.py \
  --audits "$CAMARL_ROOT"/outputs/audits/checkpoint-*.json \
  --pool-size 1000 \
  --frozen-dir "$CAMARL_ROOT/checkpoints/camarl_N1000" \
  --output "$CAMARL_ROOT/checkpoints/camarl_N1000_policy.json"
```

Repeat the audit command for every candidate checkpoint before selection.

## 8. Inference and evaluation

Serve the base model with the selected LoRA adapter, then run:

```bash
python model/scripts/eval_magrpo_coordinator_graphhard.py \
  --coordinator-model-name CAMARL_N1000 \
  --base-model-name Qwen3.5_4B \
  --coordinator-adapter "$CAMARL_ROOT/checkpoints/camarl_N1000" \
  --policy-manifest "$CAMARL_ROOT/checkpoints/camarl_N1000_policy.json" \
  --trained-pool-size 1000 \
  --pool-sizes 1000 \
  --ports 8400 \
  --checkpoint "$BASE_MODEL" \
  --data-dir "$MOSAICDIFF_ROOT/raw" \
  --graphhard-dir "$GRAPHHARD_DIR" \
  --expected-protocol-sha256 "$PROTOCOL_SHA256" \
  --media-dir "$MOSAICDIFF_ROOT/processed/topic_frames" \
  --output-dir "$CAMARL_ROOT/outputs/test_N1000" \
  --fail-on-request-error
```

Use `aggregate_magrpo_five_seeds.py` to aggregate seeds 13, 21, 34, 55,
and 89 after all evaluations finish.

## 9. Reproducibility contract

1. Split at cascade level and never tune on test cascades.
2. Build the graph and miner from training records only.
3. Freeze candidate pools before running model baselines or CAMARL.
4. Build CAMARL states only from the fixed-all validation cache.
5. Keep reward-only target fields out of model prompts.
6. Select checkpoints on validation-dev before test evaluation.
7. Preserve protocol, dataset, adapter, and result SHA-256 values.
8. Report all five decoding seeds for the complete paper protocol.

## 10. Data and model release policy

- No base-model or CAMARL model-weight file is included in Git.
- Raw MosaicDiff is distributed only through the external dataset URL.
- The two `.pt` files in the raw artifact are frozen news feature tensors, not
  CAMARL policy weights.
- Dataset redistribution rights, source-platform terms, consent/legal basis,
  privacy review, takedown contact, and final license must be completed before
  the external raw artifact is made public.
- Social profiles, histories, graph relations, text, images, and videos may
  contain personal or copyrighted material. Public release requires an explicit
  rights and privacy review rather than relying on the code's MIT license.
- CAMARL is for research on information diffusion reranking and is not intended
  for employment, credit, policing, healthcare, or other high-impact decisions.

## 11. Publication placeholders

Replace before the final release:

- `REPLACE_WITH_RAW_DATASET_DOWNLOAD_URL`
- paper DOI or arXiv URL in `CITATION.cff`
- final dataset DOI and license
- base-model revision and license
- hardware and final evaluation table

## Citation and license

Citation details are waiting for the paper release; provisional metadata is
stored in `CITATION.cff`. Source code is released under
the MIT License in `LICENSE`. The dataset and any separately released adapter
require their own licenses.
