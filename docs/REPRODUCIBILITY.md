# Reproducibility contract

1. Build splits at cascade level; never tune on test cascades.
2. Build the graph and miner from training records only.
3. Freeze candidate pools before running language-model baselines or CAMARL.
4. Build CAMARL states only from the `fixed_all` validation cache.
5. Keep reward-only target fields out of prompts.
6. Select checkpoints on validation-dev and freeze before test evaluation.
7. Report five decoding seeds (13, 21, 34, 55, 89) when reproducing the full
   paper protocol.
8. Preserve the generated audits and SHA-256 manifests with every result.

The code contains explicit checks for split provenance, target visibility,
candidate-pool hashes, request failures, and adapter identity.
