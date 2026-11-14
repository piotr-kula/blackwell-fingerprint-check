# Prompt for ASTRA

Copy everything below the line and give it to ASTRA as its task.

---

You are working on an NVIDIA DGX Spark (Linux, aarch64, GB10 GPU) through a shell. You have no
other context for this task; everything you need is here.

## What this is

A published report claims that the weights of some AI models record which type of GPU created
them, because PyTorch's random initialisation on a GPU lays numbers out according to the GPU's
thread count. The owner of this machine wants to check that mechanism on this GPU. You will run
one self-contained test script from a private repository and send back its output. The script
does not use the network and writes only inside its own folder.

You are collecting measurements. Do not interpret them beyond what the script prints, and do not
try to make a failing stage pass.

## Rules

- Do not run code from any other repository. In particular do not clone or run
  `macrogrants/checkpoint-forensics` or `bojieli/checkpoint-forensics`.
- Do not install anything system-wide. No `sudo`, no `apt`, no `pip install` outside the virtual
  environment you create inside the repository folder.
- Do not change GPU, driver, power, firmware or kernel settings.
- Do not use, create or ask for a Hugging Face token. Do not download model weights.
- Do not edit `spark_check.py` or the files in `data/`. If the script crashes, report the
  traceback as it is.
- Do not use the `--simulate-device` option. It is a developer self-test and its results mean
  nothing for hardware.
- Do not open a pull request, an issue or a release, and do not push to `main`.
- Do not create or delete files outside the repository folder.
- If a step needs something these rules forbid, stop and report what is blocking you.

## Step 1: get the kit

```bash
cd ~
git clone https://github.com/piotr-kula/blackwell-fingerprint-check.git
cd blackwell-fingerprint-check
git checkout -b results/spark-$(date -u +%Y%m%d)
```

The repository is private. Use the git credentials already configured on this machine. If the
clone fails with an authentication error, stop and report it; do not create credentials.

## Step 2: record the system

```bash
mkdir -p results
{ date -u; uname -a; cat /etc/os-release; python3 --version; } > results/system.txt 2>&1
nvidia-smi > results/nvidia-smi.txt 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> results/nvidia-smi.txt 2>&1
```

## Step 3: a Python environment with CUDA PyTorch

First see whether the system Python already has a CUDA-enabled torch. This only reads.

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**If it prints a version and `True`**, reuse it:

```bash
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -c "import numpy, torch; print(numpy.__version__, torch.__version__, torch.cuda.is_available())"
```

If numpy is missing, run `pip install numpy` (this installs into `.venv` only).

**Otherwise** make a clean environment and install torch into it. The download is 1 to 3 GB and
goes into `.venv` only.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then record what you ended up with:

```bash
pip freeze > results/pip-freeze.txt
```

What to do if this goes wrong:

| Symptom | Action |
|---|---|
| `python3 -m venv` fails because `ensurepip` or `venv` is missing | If `uv` is installed, use `uv venv .venv`. Otherwise stop and report; installing `python3-venv` needs `sudo`, which is not allowed. |
| No matching wheel for `torch` on the `cu130` index | Keep the error text. Try the same install with `cu129` in the URL, then `cu128`. Report which one worked. |
| torch installs but `torch.cuda.is_available()` is `False` | Check that `nvidia-smi` works. Report the output of `python -c "import torch; print(torch.__config__.show())"`. Then go on to step 4 anyway: the script will run its CPU stage and say the GPU stages were skipped. |
| A warning that the GPU's capability `sm_121` is not supported | Not an error. Continue. The script records the warning. |
| Docker with an NVIDIA PyTorch image is the only way to get a CUDA torch | Allowed only if `docker` already works without `sudo`. Run the script inside the container with this folder mounted, and report the exact image name and digest. Do not install Docker. |

## Step 4: run the main test

```bash
python spark_check.py 2>&1 | tee results/stdout.txt
echo "exit code: ${PIPESTATUS[0]}" | tee -a results/stdout.txt
cp results.json results/results.json
```

It should take a few minutes. Exit code 0 means every stage that ran passed. Exit code 1 means at
least one stage printed `FAIL` or `ERROR`; that is a result, not a crash, and you still report it.

| What you see | Action |
|---|---|
| `[A] FAIL` | Stop. The kit or numpy is broken. Report the output. |
| Stages B to D print `SKIPPED` with a "no CUDA" reason | torch cannot see the GPU. Go back to step 3. If you cannot fix it within the rules, report as is. |
| `[C] FAIL` | Report as is. Do not change anything. |
| `[C+] SKIPPED` or `ERROR`, or an out-of-memory error | Run once more with `python spark_check.py --skip-large 2>&1 | tee results/stdout-skip-large.txt` and keep both outputs. |
| `[D] FAIL` | This can be a genuine finding. Report as is. |
| The script itself raises a traceback and stops | Report the traceback. `results.json` is written after every stage, so copy whatever exists. |

## Step 5: the optional blind search (only if stages A, B and C printed PASS)

This is a second, separate measurement. Report it separately.

First a short slice that should find a seed, to check that it works and to see the speed:

```bash
python spark_check.py --blind-search --seed-range 952000000:954097152 --restart --out results_blind_slice.json 2>&1 | tee results/stdout-blind-slice.txt
```

It prints its throughput and an estimate for the full run. If that estimate is under 3 hours,
run the full search:

```bash
python spark_check.py --blind-search --restart 2>&1 | tee results/stdout-blind.txt
echo "exit code: ${PIPESTATUS[0]}" | tee -a results/stdout-blind.txt
cp results_blind.json results_blind_slice.json results/
```

It prints progress and an ETA every 10 seconds and saves a checkpoint every 30 seconds. If it is
interrupted, run the same command **without** `--restart` and it resumes. If the estimate is over
3 hours, do not run the full search; copy `results_blind_slice.json` and report the estimate.

## Step 6: commit the results

```bash
git add results/
git commit -m "DGX Spark results $(date -u +%Y-%m-%d)"
git push -u origin HEAD
```

Commit only the `results/` folder, on the branch you created in step 1. Push that branch and
nothing else.

## Step 7: what to send back

1. The branch name and commit hash.
2. The full contents of `results/stdout.txt`.
3. The contents of `results/nvidia-smi.txt`.
4. The `=== Summary ===` line and every `[X] PASS/FAIL/SKIPPED` line, repeated at the top of your
   message.
5. From stage B: torch version, device name, compute capability, SM count, threads per SM.
6. From stage C: the thread count `T` it found and the runner-up.
7. If you ran step 5: the `RECOVERED` line or the "no seed" line, the runner-up line, the
   throughput line and the wall time, under a separate heading.
8. Anything you had to do that is not in these instructions, and anything you could not do.

Send the files as they are. Do not summarise numbers from memory.
