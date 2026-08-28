---
language:
  - en
library_name: transformers
pipeline_tag: text-generation
license: other
tags:
  - reinforcement-learning
  - multi-agent
  - reranking
  - information-diffusion
---

# CAMARL model card

CAMARL is a cost-aware evidence-acquisition coordinator for next-participant
reranking in information cascades. It chooses among semantic, profile, and
topology experts, or stops and emits a candidate ranking.

## Intended use

- Research on information diffusion reranking.
- Auditing adaptive expert acquisition and ranking/cost trade-offs.
- Reproduction of the CAMARL paper under the released protocol.

The model is not intended for decisions about employment, credit, policing,
healthcare, or other high-impact uses. It predicts behavior from historical
social data and can reproduce dataset bias.

## Training

- Base model: Qwen3.5-4B-compatible checkpoint supplied separately.
- Adaptation: LoRA with rank 16, alpha 32, dropout 0.05.
- Optimization: group-relative reinforcement learning.
- Selected release configuration group size: 4.
- Reward weights: ranking utility 0.65, reference-action alignment 0.30,
  valid-format reward 0.05.
- Acquisition penalty: 0.10; invalid-format reward: -0.50.
- Training data: validation-only coordinator states; test labels are rejected.

No model weights are bundled with this repository. Every separately published
adapter should include its generated `training_audit.json` and policy manifest;
those files are authoritative for that adapter.

## Limitations

- Requires the accompanying candidate-pool protocol and expert evidence.
- Results depend on the base checkpoint, decoding stack, and social-data scope.
- User descriptions and histories may be sparse, noisy, outdated, or biased.
- The release does not grant rights to the base model or dataset.

## Artifact placeholders

Replace before publication:

- Adapter download URL
- Base-model revision and license
- Dataset DOI/repository URL: `REPLACE_WITH_CAMARL_DATASET_DOWNLOAD_URL`
- Paper DOI/arXiv URL
- Evaluation table and hardware details
