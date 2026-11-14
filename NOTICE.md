# Notice and attribution

**Code.** `spark_check.py` was written for this review. It contains no code from
`macrogrants/checkpoint-forensics` or `bojieli/checkpoint-forensics` (both MIT). The method it
tests was described by Will DePue; the replication by Bojie Li was read during the review.

**Algorithms and constants.**
- Philox4x32-10: J. Salmon, M. Moraes, R. Dror, D. Shaw, "Parallel Random Numbers: As Easy as
  1, 2, 3", SC'11. The three known-answer vectors are the published Random123 test vectors.
- The mapping from (seed, thread, offset) to Philox counter and key, and the Box-Muller step,
  follow the documented behaviour of NVIDIA cuRAND's device API (`curand_init`, `curand_normal4`).
  No NVIDIA header text is reproduced.
- The thread layout follows PyTorch (BSD-style licence),
  `aten/src/ATen/native/cuda/DistributionTemplates.h`.

**Data.** `data/v4flash_rows_u16le.bin` holds 36 rows of the `embed.weight` tensor of
DeepSeek-V4-Flash, fetched with HTTP range requests from
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash at revision
`60d8d70770c6776ff598c94bb586a859a38244f1`. The exact file, byte ranges and SHA-256 are in
`data/PROVENANCE.json`. DeepSeek publishes the model under the MIT licence, reproduced in
`data/LICENSE-DeepSeek.txt` (Copyright (c) 2023 DeepSeek).
