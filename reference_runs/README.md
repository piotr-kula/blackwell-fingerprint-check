# Reference runs from the review PC

These are actual outputs of `spark_check.py`, produced on the review PC on 2026-11-13: Windows 11,
AMD Ryzen 9 9950X3D (16 cores), no NVIDIA GPU, Python 3.12.10, numpy 2.5.4, torch 2.15.0 (CPU
build, used only for the blind search). None of them is a GPU result.

| File | What it is |
|---|---|
| `stage_A_stdout.txt`, `stage_A_results.json` | `python spark_check.py --stages A`. Real result: PASS. |
| `blind_full_cpu_stdout.txt`, `blind_full_cpu_results.json` | `python spark_check.py --blind-search --restart` on the CPU with torch: all 2^32 seeds under five thread counts in 630 s (34.1 million seed-layout evaluations per second). Real result: one hit, seed 953,049,583 at T = 303,104, held-out r = 0.999999; the runner-up matched 2 of 32 probe values. |
| `SIMULATED_not_hardware_T73728_stdout.txt` | `python spark_check.py --simulate-device 73728`. A code self-test in which the emulator stands in for the GPU. It shows what a full run prints. Stage D fails there by construction, because the stand-in uses correctly rounded math. It says nothing about any hardware. |
