# blackwell-fingerprint-check: a DGX Spark test kit

This kit tests, on an NVIDIA GPU you control, the mechanism behind Will DePue's report
"Top Chinese Frontier AI Models Carry the Fingerprint of Export-Controlled NVIDIA Blackwell GPUs"
(Unemployment Labs, 2026-11-12). It was written during an adversarial review of that report and of
Bojie Li's replication. It contains one script, `spark_check.py`, a 288 KB sample of public model
weights, and instructions for the agent that runs it.

It needs Python 3 with numpy, and PyTorch with CUDA for the GPU stages. It does not use the
network, does not need a Hugging Face token, and writes only inside this folder.

## The claim, in layers

The review split the claim into five steps. The kit only reaches the first two.

| Layer | Statement | Review verdict | What this kit adds |
|---|---|---|---|
| (a) | Rows of the public weights equal PyTorch's CUDA random fill for a seed S and a thread count T | Holds. Reproduced with independent code on rows fetched independently. | Stages A, C, D and F test it again, on real NVIDIA hardware. |
| (b) | T identifies the chip family (303,104 = B200/B300, 311,296 = GB200/GB300) | Holds with caveats. 148 SMs for B200 is public; no public figure found for 152. AMD at half size gives 311,296. | Stage C tests the formula T = SMs x blocks per SM x 256 on one more device. Stage E shows T is whatever the software writes. |
| (c) | The chip that created the weights also trained the model | Not established by the weights. | Nothing. |
| (d) | Export controls were violated, or the chips were in China | Unsupported. The report itself disclaims it. | Nothing. |
| (e) | The GPU-count estimates | Unsupported by any published code. | Nothing. |

A DGX Spark cannot tell you how many SMs a B200 or GB200 reports. That takes a few minutes on
the real hardware: print `torch.cuda.get_device_properties(0).multi_processor_count`.

## The mechanism being tested

PyTorch fills a tensor on a GPU with `T` threads, where

    grid.x = min(multiProcessorCount * (maxThreadsPerMultiProcessor / 256), ceil(numel / 256))
    T      = 256 * grid.x

Element `i` is written by thread `i % T`. Each thread has its own Philox stream (the thread index
is the Philox subsequence, the generator's seed is the key), and each `curand_normal4` call yields
four values that land `T` apart. The same seed on a device with a different `T` produces the same
numbers in different places.

Source: `aten/src/ATen/native/cuda/DistributionTemplates.h`, functions `calc_execution_policy`
and `distribution_elementwise_grid_stride_kernel`, read at PyTorch main
`0a95a45a42fbec1ab15a5c8ad9b6f03ac2e225ef` (2026-11-12). The grid formula is the same in
v1.3.0, v1.8.0, v2.0.0, v2.5.0 and v2.9.0.

## Stages

Run `python spark_check.py`. Each stage prints `PASS`, `FAIL` or `SKIPPED` and everything goes
into `results.json`.

**A. CPU self-test (no GPU).** Checks the Philox implementation against the three Random123
known-answer vectors. Then regenerates the 36 bundled rows of DeepSeek-V4-Flash's embedding from
seed 953049583 at T = 303,104 and checks that T = 270,336, T = 311,296 and seed +/- 1 do not
reproduce them. Measured on the review PC: 36/36 rows at r = 0.999999; the four controls at
|r| <= 0.044; 147,416 of 147,456 bf16 values bit-identical to correctly rounded math.

**B. Device facts.** torch and CUDA versions, device name, compute capability, SM count, and
threads per SM when torch exposes `max_threads_per_multi_processor` (torch 2.1 and later do).

**C. Mechanism on this GPU.** With stock PyTorch and fixed seeds:
- sweeps every grid size from 1 to 2,600 blocks and reports the unique `T` that reproduces
  torch's own `normal_()` output, and the runner-up;
- checks `T` against the formula above;
- counts float32 bit agreement with the emulator;
- checks bfloat16 and float16 fills, `std` scaling, a 2-D tensor, tensors smaller than `T`, and
  two fills in a row (the second starts at the Philox offset the first consumed);
- confirms the 132, 148 and 152 x 2048 layouts do not match this device's output.

A separate line, `C+`, tests the review's derivation of how tensors larger than 2^31 bytes are
filled (as halves, at offsets that depend on `T`). That derivation explains the unexplained
offset 3776 in Kimi-K3; it was confirmed on public weights but never on a GPU. It allocates
2.2 GiB; skip it with `--skip-large`.

**D. NVIDIA device math.** The emulator rounds `log`, `sin` and `cos` correctly. Real GPU float32
math does not, in a small fraction of cases.
- D1 measures how often this GPU differs from correct rounding, and whether the differences sit
  where `sin` or `cos` is near zero. For comparison, the public weights differ from correctly
  rounded math in 2.45e-4 of bf16 values for V4-Flash, 2.51e-4 for V4-Pro and 2.63e-4 for V3,
  about 60% of them near such zeros.
- D2 reproduces the bundled public rows with stock PyTorch on this GPU. A value written by thread
  `t` at step `q` of a 303,104-thread fill is the same Philox call as position `q * T_dev + t` of a
  fill on this device, provided `t < T_dev`. The bundled rows were chosen so that `t < 73,728`.
  The script generates one bfloat16 tensor (about 245 MiB if `T_dev` is 73,728), reads those
  positions and compares bits. Of the 147,456 bundled values, 40 are ones where correctly rounded
  math disagrees with the public weights; D2 reports how many of those this GPU gets right.

**E. Spoofing demo.** Writes a tensor in the 303,104 and 311,296 layouts in software, places it on
this GPU, and shows the layout reader reports 148 and 152 SMs. It demonstrates that the layout is
a property of what the software wrote. It does not show that anyone did this.

**F. Blind seed recovery (optional, `--blind-search`).** Given only the bundled rows, searches all
2^32 seeds under several thread counts (132, 148, 152, 108 and a made-up 144, each x 2048, plus this
device's own) and reports the seed and thread count it finds, with the runner-up. It estimates the
scale from the data, tests each seed against the 32 largest values of the first four rows, and
confirms on the other 32 rows, which play no part in the search. It checkpoints to
`blind_search_checkpoint.json` and resumes if run again. `--seed-range LO:HI` runs a slice.

Measured on the review PC (AMD Ryzen 9 9950X3D, CPU only): the full 2^32 range under five thread
counts took 630 s with torch on the CPU, 34.1 million seed-layout evaluations per second, and
found exactly one hit: seed 953,049,583 at T = 303,104, held-out r = 0.999999. The runner-up
matched 2 of the 32 probe values. With numpy alone the rate was 6.0 million per second, about
one hour for the same search.

Estimate for the DGX Spark, not measured: 5 to 40 minutes for six thread counts with torch on the
GPU. The search is plain elementwise tensor arithmetic, so its speed depends on memory bandwidth,
which was not measured. The script prints its own measured rate and ETA after ten seconds.

## How to read the outcomes

| Outcome | Reading |
|---|---|
| A fails | The kit or the numpy install is broken. Nothing else means anything. |
| C passes with T = SMs x (threads per SM // 256) x 256 | The formula holds on this device. Layer (a)'s mechanism is confirmed on a third device family. |
| C finds a unique T that the formula does not predict | Layer (b) is weaker than assumed: device properties do not determine T the way the source says. Report it. |
| C finds no unique T | This torch build does not use the grid-stride Philox fill, or the emulator is wrong. Report versions. |
| C+ fails | The review's large-tensor derivation is wrong, although it fits Kimi-K3 and A.X-K2. |
| D2: all values reproduced, including the 40 | The public rows are bit-for-bit what NVIDIA device math produces. This supports the report's claim that the V4-Flash values point to NVIDIA rather than AMD. It rests on device math being the same across NVIDIA GPUs and CUDA versions; a full match also validates that. |
| D2: this GPU agrees with correct rounding on the 40 | This GPU and torch build do not produce the deviations in the public weights. Ambiguous: a different CUDA math library version, or the weights came from other math. |
| D1 rate near 2.5e-4, concentrated at zeros | Same conclusion as a D2 pass, statistically. |
| E passes | Expected. Any device can hold any layout. |
| F recovers 953049583 at 303,104 with no competitor | Seed and layout are identifiable from the weights alone, with no hint. |
| F finds nothing over the full range | The search assumptions (offset 0, these layouts, seed below 2^32) do not hold for these rows. |

## What the kit does not show

- That a B200, B300, GB200 or GB300 has the SM count the report assigns to it.
- Anything about where a model was trained, on what, by whom, or under what licence.
- Anything about models other than DeepSeek-V4-Flash. The review checked five others with the
  same emulator; their rows are not bundled.
- Stage F is told neither seed nor layout, but the bundled rows were picked by the reviewer, who
  knew which rows are untouched. It also searches generator offset 0 only.

## What was tested before shipping, and what was not

Tested on the review PC (Windows, no NVIDIA GPU, numpy 2.5.4, torch 2.15.0 CPU):
- Stage A: real run, PASS.
- Stages C, D and E: run end to end against a simulated device (`--simulate-device 73728`, the
  emulator standing in for the GPU). This tests the code paths and the D2 index mapping. It says
  nothing about hardware.
- The torch calls in the device wrapper: run on the CPU.
- torch installed but no CUDA: stages B to D skip, E runs on the CPU.
- Stage F: slices with and without the true seed, on numpy and on torch CPU; interrupt and
  resume; and the full 2^32 range on torch CPU (see `reference_runs/`).

Not tested: stages B, C, D and F on any CUDA device.

Assumptions about the DGX Spark that could not be confirmed:
- 48 SMs (from a Chips and Cheese article, not a device query).
- 1,536 threads per SM, which would give T = 73,728. If it is 2,048, T = 98,304. Stage C finds
  the value either way.
- That PyTorch's Linux aarch64 CUDA wheels (index `cu130`, versions 2.9.0 to 2.14.1 at the time
  of writing) run on the GB10's compute capability 12.1.

## Files

| File | Purpose |
|---|---|
| `spark_check.py` | The test. `--help` lists the options. |
| `data/v4flash_rows_u16le.bin` | 36 rows x 4,096 bfloat16 values of DeepSeek-V4-Flash's `embed.weight`. |
| `data/PROVENANCE.json` | Source repository, revision, file, byte ranges and SHA-256 of the sample. |
| `data/LICENSE-DeepSeek.txt` | The MIT licence under which DeepSeek publishes those weights. |
| `ASTRA_PROMPT.md` | Instructions for the agent that runs the kit on the Spark. |
| `RESULTS_TEMPLATE.md` | What a filled-in report looks like. |
| `reference_runs/` | Actual output from the review PC (CPU only). |
| `NOTICE.md` | Attribution. |
| `requirements.txt` | numpy, and torch 2.1 or later with CUDA for the GPU stages. |

## Sources

- Report: Will DePue, https://unemploymentlabs.com/checkpoint-forensics (2026-11-12).
- DePue's code: https://github.com/macrogrants/checkpoint-forensics at commit
  `b633870c4a8b0f85d70026b7f77f7d0858358f89`.
- Li's replication: https://github.com/bojieli/checkpoint-forensics at commit
  `644d8172584998ad9e201ce32ff13d59f140f1e0`; write-up https://01.me/en/2026/11/checkpoint-forensics/.
- PyTorch: `aten/src/ATen/native/cuda/DistributionTemplates.h`,
  `aten/src/ATen/cuda/CUDAGeneratorImpl.cpp`, `aten/src/ATen/TensorIterator.cpp`
  (`can_use_32bit_indexing`, `SplitUntil32Bit`) at `0a95a45a42fbec1ab15a5c8ad9b6f03ac2e225ef`.
- Weights: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash at revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`.

No code from either reviewed repository is included. The emulator was written from the Philox
paper, the documented behaviour of cuRAND's device API and the PyTorch source above.
