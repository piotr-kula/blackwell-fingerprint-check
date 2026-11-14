#!/usr/bin/env python3
"""spark_check.py - test kit for the "GPU fingerprint in model weights" mechanism on an NVIDIA GPU.

What it does (details in README.md):
  A  CPU self-test: Philox4x32-10 known answers; reproduce shipped rows of the public
     DeepSeek-V4-Flash embedding from seed 953049583 at T = 303,104 threads, and show that
     T = 270,336, T = 311,296 and seed +/- 1 do not reproduce them.
  B  Device facts (torch / CUDA versions, GPU name, SM count, threads per SM if exposed).
  C  Mechanism test on this GPU with stock PyTorch: which thread count T reproduces
     torch's own CUDA normal_() fill, and does it equal PyTorch's formula?
  D  Is NVIDIA's device math bit-identical to correctly rounded math? Where does it deviate?
     Does this GPU reproduce the public rows bit for bit?
  E  Spoofing demo: software can write any layout on any device.
  F  Optional, only with --blind-search: recover the seed and the thread count from the shipped
     rows alone by brute force over all 2**32 seeds (checkpointed, resumable).

Only needs numpy (stage A) and torch with CUDA (stages B-E). No network access, no Hugging
Face token, no files written outside this folder. The emulator below is the reviewer's own
implementation written from the Philox paper (Salmon et al., SC'11), the documented cuRAND
device-API semantics and PyTorch's aten/src/ATen/native/cuda/DistributionTemplates.h.
No code from the reviewed repositories is used.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
import warnings

import numpy as np

KIT_VERSION = "1.0"
KIT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_SOURCE = ("pytorch/pytorch aten/src/ATen/native/cuda/DistributionTemplates.h, calc_execution_policy() and "
                  "distribution_elementwise_grid_stride_kernel(); read at main 0a95a45a42fbec1ab15a5c8ad9b6f03ac2e225ef "
                  "(2026-11-12); same grid formula in v1.3.0, v1.8.0, v2.0.0, v2.5.0, v2.9.0")
FORMULA = ("grid.x = min(multiProcessorCount * (maxThreadsPerMultiProcessor / 256), ceil(numel / 256)); block = 256; "
           "T = 256 * grid.x; element i is written by thread i % T (Philox subsequence = thread index), "
           "lane (i // T) % 4 of curand_normal4 call number (i // T) // 4 (Philox counter = generator_offset/4 + call)")

# ----------------------------------------------------------------------------------------------
# Emulator (numpy)
# ----------------------------------------------------------------------------------------------
_M0, _M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_MASK, _S32 = np.uint64(0xFFFFFFFF), np.uint64(32)
F32, F64 = np.float32, np.float64

KAT = [  # Random123 known-answer vectors for philox4x32, 10 rounds: (counter, key, expected)
    ((0x00000000, 0x00000000, 0x00000000, 0x00000000), (0x00000000, 0x00000000),
     (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 0xFFFFFFFF),
     (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


def philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10. Counter words are uint64 arrays holding 32-bit values; key words are Python ints."""
    c0, c1, c2, c3 = (np.asarray(x, dtype=np.uint64) & _MASK for x in (c0, c1, c2, c3))
    k0, k1 = int(k0) & 0xFFFFFFFF, int(k1) & 0xFFFFFFFF
    for _ in range(10):
        p0 = _M0 * c0  # both factors < 2**32, so the uint64 product is exact
        p1 = _M1 * c2
        n0 = (p1 >> _S32) ^ c1 ^ np.uint64(k0)
        n2 = (p0 >> _S32) ^ c3 ^ np.uint64(k1)
        c0, c1, c2, c3 = n0, p1 & _MASK, n2, p0 & _MASK
        k0, k1 = (k0 + _W0) & 0xFFFFFFFF, (k1 + _W1) & 0xFFFFFFFF
    return c0, c1, c2, c3


_INV32 = F32(2.3283064e-10)                 # CURAND_2POW32_INV, exactly 2**-32
_TWO_PI32 = F32(6.2831855)
_INV_2PI32 = F32(_INV32 * _TWO_PI32)         # CURAND_2POW32_INV_2PI (exact power-of-two scaling)
_HALF_INV32 = F32(_INV32 / F32(2))
_HALF_INV_2PI32 = F32(_INV_2PI32 / F32(2))


def normal_core(li, T, seed, offset=0, fma=True, want_aux=False):
    """Standard-normal float32 values that curand_normal4 yields for linear indices `li` of a fill that
    uses T threads in total, generator seed `seed` and Philox offset `offset` (multiple of 4).

    Math functions are evaluated in float64 and rounded once to float32, i.e. this is the *correctly
    rounded* reference. A real GPU uses NVIDIA's float32 device math, which may differ in the last bits.
    fma=True models contraction of y*c + c/2 into one fused multiply-add (one rounding)."""
    if offset % 4:
        raise ValueError("offset must be a multiple of 4")
    li = np.asarray(li, dtype=np.uint64)
    T64 = np.uint64(T)
    t = li % T64
    q = li // T64
    lane = (q % np.uint64(4)).astype(np.int64)
    call = q // np.uint64(4) + np.uint64(offset // 4)
    seed = int(seed)
    w = philox4x32_10(call & _MASK, call >> _S32, t & _MASK, t >> _S32, seed & 0xFFFFFFFF, seed >> 32)
    lo = lane < 2
    x = np.where(lo, w[0], w[2]).astype(F64).astype(F32)   # uint32 -> float32, one rounding
    y = np.where(lo, w[1], w[3]).astype(F64).astype(F32)
    u = (x * _INV32 + _HALF_INV32).astype(F32)             # product is exact (power of two)
    if fma:
        v = (y.astype(F64) * F64(_INV_2PI32) + F64(_HALF_INV_2PI32)).astype(F32)
    else:
        v = ((y * _INV_2PI32).astype(F32) + _HALF_INV_2PI32).astype(F32)
    s = np.sqrt((F32(-2.0) * np.log(u.astype(F64)).astype(F32)).astype(F32)).astype(F32)
    is_sin = (lane & 1) == 0
    v64 = v.astype(F64)
    trig64 = np.where(is_sin, np.sin(v64), np.cos(v64))
    z = (trig64.astype(F32) * s).astype(F32)
    if want_aux:
        return z, {"lane": lane, "u": u, "v": v, "s": s, "trig": trig64, "is_sin": is_sin}
    return z, None


def f32_to_bf16_bits(a):
    """float32 -> bfloat16 bit pattern, round to nearest even (as torch does)."""
    b = np.ascontiguousarray(a, dtype=F32).view(np.uint32)
    return ((b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(bits):
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(F32)


def to_bits(vals_f32, dtype):
    vals_f32 = np.ascontiguousarray(vals_f32, dtype=F32)
    if dtype == "float32":
        return vals_f32.view(np.uint32).copy()
    if dtype == "bfloat16":
        return f32_to_bf16_bits(vals_f32)
    if dtype == "float16":
        with np.errstate(over="ignore"):
            return vals_f32.astype(np.float16).view(np.uint16).copy()
    raise ValueError(dtype)


def bits_to_f64(bits, dtype):
    if dtype == "float32":
        return np.ascontiguousarray(bits, dtype=np.uint32).view(F32).astype(F64)
    if dtype == "bfloat16":
        return bf16_bits_to_f32(bits).astype(F64)
    if dtype == "float16":
        return np.ascontiguousarray(bits, dtype=np.uint16).view(np.float16).astype(F64)
    raise ValueError(dtype)


ITEMSIZE = {"float32": 4, "bfloat16": 2, "float16": 2}
INT32_MAX = 2 ** 31 - 1


def eff_threads(numel, T_dev):
    """T actually used for a tensor of numel elements on a device whose full grid is T_dev threads."""
    return min(T_dev // 256, (numel + 255) // 256) * 256


def offset_inc(numel, T_eff):
    """How far one fill advances the generator's Philox offset (counter_offset in calc_execution_policy)."""
    return ((numel - 1) // (T_eff * 4) + 1) * 4


def can_use_32bit(numel, itemsize):
    """TensorIteratorBase::can_use_32bit_indexing for a contiguous 1-D view (byte offsets must fit int32)."""
    return numel <= INT32_MAX and 1 + (numel - 1) * itemsize <= INT32_MAX


def fill_plan(numel, itemsize, T_dev, base_offset=0):
    """Segments (start, length, philox_offset, T_eff) of one normal_() call and the generator offset afterwards.

    A tensor that cannot use 32-bit indexing (> 2**31 bytes) is filled as halves, recursively. The outer call
    consumes the full-tensor offset increment *before* splitting; each leaf then consumes its own increment
    and uses indices local to the leaf. (Reviewer's reading of distribution_nullary_kernel + SplitUntil32Bit.)"""
    T_top = eff_threads(numel, T_dev)
    if can_use_32bit(numel, itemsize):
        return [(0, numel, base_offset, T_top)], base_offset + offset_inc(numel, T_top)
    off = base_offset + offset_inc(numel, T_top)
    leaves, stack = [], [(0, numel)]
    while stack:
        start, length = stack.pop()
        if can_use_32bit(length, itemsize) or length < 2:
            leaves.append((start, length))
        else:
            first = length // 2
            stack.append((start + first, length - first))
            stack.append((start, first))
    plan = []
    for start, length in leaves:
        T = eff_threads(length, T_dev)
        plan.append((start, length, off, T))
        off += offset_inc(length, T)
    return plan, off


def emulate_at(pos, plan, seed, fma=True, chunk=1 << 20):
    """Correctly rounded standard-normal float32 values at global positions `pos` for a fill described by `plan`."""
    pos = np.asarray(pos, dtype=np.int64)
    out = np.empty(pos.shape, dtype=F32)
    starts = np.array([p[0] for p in plan], dtype=np.int64)
    seg = np.searchsorted(starts, pos, side="right") - 1
    for k, (start, _length, off, T) in enumerate(plan):
        idx = np.nonzero(seg == k)[0]
        for a in range(0, idx.size, chunk):
            j = idx[a:a + chunk]
            out[j] = normal_core(pos[j] - start, T, seed, off, fma)[0]
    return out


def scale_f32(z, std):
    """transformation::normal with mean 0: float32(rand * float32(std)) (an FMA with addend 0 is the same)."""
    if std == 1.0:
        return np.ascontiguousarray(z, dtype=F32)
    return (np.ascontiguousarray(z, dtype=F32) * F32(std)).astype(F32)


def pearson(a, b):
    a = np.asarray(a, dtype=F64).ravel()
    b = np.asarray(b, dtype=F64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 0 else 0.0


def ulp_distance(a_bits, b_bits):
    """Distance in float32 units-in-the-last-place between two arrays of float32 bit patterns."""
    def mono(u):
        i = np.ascontiguousarray(u, dtype=np.uint32).astype(np.int64)
        neg = i >= 0x80000000
        return np.where(neg, -(i & 0x7FFFFFFF), i)
    return np.abs(mono(a_bits) - mono(b_bits))


def sweep_layout(sample_pos, sample_vals, seed, offset, g_lo=1, g_hi=2600, fma=True):
    """Correlate observed values at sample positions with the emulator for every grid size g (T = 256 g)."""
    res = []
    for g in range(g_lo, g_hi + 1):
        z = normal_core(sample_pos, 256 * g, seed, offset, fma)[0]
        res.append((pearson(sample_vals, z), g))
    res.sort(reverse=True)
    return res


# ----------------------------------------------------------------------------------------------
# Devices
# ----------------------------------------------------------------------------------------------
class TorchCudaDevice:
    simulated = False

    def __init__(self, torch):
        self.torch = torch
        self.device = "cuda"
        self.seed_fn = torch.cuda.manual_seed

    def fill(self, numel, dtype, seed, std=1.0, positions=None, reseed=True, shape=None):
        """Stock PyTorch: seed the CUDA generator, call normal_() once, return bit patterns (all or gathered)."""
        torch = self.torch
        if reseed:
            self.seed_fn(int(seed))
        t = torch.empty(shape if shape is not None else int(numel), dtype=getattr(torch, dtype), device=self.device)
        t.normal_(0.0, float(std))
        flat = t.reshape(-1)
        if positions is not None:
            idx = torch.as_tensor(np.asarray(positions, dtype=np.int64), device=self.device)
            flat = flat[idx]
        flat = flat.contiguous()
        if dtype == "float32":
            bits = flat.view(torch.int32).cpu().numpy().view(np.uint32).copy()
        else:
            bits = flat.view(torch.int16).cpu().numpy().view(np.uint16).copy()
        del t, flat
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return bits

    def free_bytes(self):
        try:
            return int(self.torch.cuda.mem_get_info()[0])
        except Exception:
            return None


class SimulatedDevice:
    """Developer self-test only: the emulator stands in for the GPU. Says nothing about any hardware."""
    simulated = True

    def __init__(self, T_dev):
        self.T, self.offset, self.seed = int(T_dev), 0, 0

    def fill(self, numel, dtype, seed, std=1.0, positions=None, reseed=True, shape=None):
        if shape is not None:
            numel = int(np.prod(shape))
        if reseed:
            self.offset, self.seed = 0, int(seed)
        plan, self.offset = fill_plan(int(numel), ITEMSIZE[dtype], self.T, self.offset)
        pos = np.arange(numel, dtype=np.int64) if positions is None else np.asarray(positions, dtype=np.int64)
        return to_bits(scale_f32(emulate_at(pos, plan, self.seed), std), dtype)

    def free_bytes(self):
        return None


# ----------------------------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------------------------
def say(msg=""):
    print(msg, flush=True)


def verdict_line(stage, status, text):
    say(f"[{stage}] {status}: {text}")


def load_sample():
    data_dir = os.path.join(KIT_DIR, "data")
    with open(os.path.join(data_dir, "PROVENANCE.json"), "r", encoding="utf-8") as f:
        prov = json.load(f)
    with open(os.path.join(data_dir, "v4flash_rows_u16le.bin"), "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != prov["sha256"] or len(raw) != prov["bytes"]:
        raise RuntimeError("row sample does not match data/PROVENANCE.json (sha256 or size)")
    d = int(prov["row_width"])
    rows = np.array(prov["rows"], dtype=np.int64)
    bits = np.frombuffer(raw, dtype="<u2").astype(np.uint16).reshape(len(rows), d)
    return prov, rows, bits


V4F = {"seed": 953049583, "T": 303104, "std": 0.006}
REFERENCE_FLIP_RATES = {  # reviewer's measurements on public weights vs correctly rounded math (bf16 values, std 0.006)
    "DeepSeek-V4-Flash (445 untouched rows)": {"flips": 446, "values": 1822720, "near_zero_share": round(268 / 446, 3)},
    "DeepSeek-V4-Pro (256 untouched rows)": {"flips": 460, "values": 1835008, "near_zero_share": round(282 / 460, 3)},
    "DeepSeek-V3 (256 untouched rows, lab-disclosed H800)": {"flips": 482, "values": 1835008, "near_zero_share": round(280 / 482, 3)},
}


# ----------------------------------------------------------------------------------------------
# Stage A - CPU self-test
# ----------------------------------------------------------------------------------------------
def stage_a():
    out = {"checks": {}}
    ok = True
    for c, k, want in KAT:
        got = tuple(int(x) for x in philox4x32_10(*c, *k))
        ok &= got == want
    out["checks"]["philox_known_answers"] = {"vectors": len(KAT), "pass": bool(ok)}
    say(f"  Philox4x32-10 known-answer vectors (Random123): {'ok' if ok else 'MISMATCH'}")

    prov, rows, bits = load_sample()
    d = bits.shape[1]
    Y = bf16_bits_to_f32(bits).astype(F64)
    li = (rows[:, None] * d + np.arange(d, dtype=np.int64)[None, :])
    out["sample"] = {"source": prov["source"], "revision": prov["revision"], "file": prov["file"], "tensor": prov["tensor"],
                     "rows": int(len(rows)), "row_width": d, "sha256": prov["sha256"],
                     "byte_ranges_inclusive": prov["byte_ranges_inclusive"]}
    say(f"  row sample: {len(rows)} rows x {d} of {prov['tensor']} from {prov['source']} @ {prov['revision'][:10]} (sha256 ok)")

    def per_row_r(T, seed):
        z = normal_core(li.ravel(), T, seed, 0)[0].reshape(li.shape)
        return np.array([pearson(Y[i], z[i]) for i in range(len(rows))]), z

    configs = [("claimed: seed 953049583, T=303,104 (148 SMs x 2048)", V4F["T"], V4F["seed"], True),
               ("T=270,336 (132 x 2048)", 270336, V4F["seed"], False),
               ("T=311,296 (152 x 2048)", 311296, V4F["seed"], False),
               ("seed+1", V4F["T"], V4F["seed"] + 1, False),
               ("seed-1", V4F["T"], V4F["seed"] - 1, False)]
    res = []
    z_claimed = None
    for label, T, seed, should_match in configs:
        r, z = per_row_r(T, seed)
        if should_match:
            z_claimed = z
            good = bool(r.min() > 0.9999)
        else:
            good = bool(np.abs(r).max() < 0.1)
        ok &= good
        res.append({"config": label, "T": T, "seed": seed, "expected_to_match": should_match, "rows_r_gt_0.99": int((r > 0.99).sum()),
                    "median_r": float(np.median(r)), "min_r": float(r.min()), "max_abs_r": float(np.abs(r).max()), "pass": good})
        say(f"  {label:52s} rows r>0.99: {int((r > 0.99).sum()):2d}/{len(rows)}  median r {np.median(r):+.6f}  -> {'ok' if good else 'UNEXPECTED'}")
    out["checks"]["layout_and_seed"] = res

    emu_bits = f32_to_bf16_bits(scale_f32(z_claimed, V4F["std"]))
    mism = emu_bits != bits
    frac = 1.0 - float(mism.mean())
    good = frac >= 0.999
    ok &= good
    out["checks"]["bf16_bit_agreement_with_correctly_rounded_math"] = {
        "values": int(bits.size), "identical": int(bits.size - mism.sum()), "differing": int(mism.sum()),
        "fraction_identical": frac, "pass": bool(good),
        "note": "the few differing values are where real GPU float32 math is not correctly rounded; stage D tests this"}
    say(f"  bf16 bit agreement with correctly rounded math: {bits.size - int(mism.sum())}/{bits.size} "
        f"({100 * frac:.4f}%), {int(mism.sum())} differing -> {'ok' if good else 'UNEXPECTED'}")
    out["status"] = "PASS" if ok else "FAIL"
    return out


# ----------------------------------------------------------------------------------------------
# Stage B - device facts
# ----------------------------------------------------------------------------------------------
def stage_b(torch, caught_warnings):
    out = {"python": sys.version.split()[0], "numpy": np.__version__, "machine": platform.machine(),
           "platform": platform.platform(), "torch": getattr(torch, "__version__", None),
           "torch_cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
           "cuda_available": bool(torch.cuda.is_available())}
    if not out["cuda_available"]:
        out["status"] = "SKIPPED"
        out["reason"] = "torch is installed but torch.cuda.is_available() is False"
        return out
    p = torch.cuda.get_device_properties(0)
    try:
        out["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception:
        out["arch_list"] = None
    out.update({
        "device_count": int(torch.cuda.device_count()), "device_index": 0,
        "device_name": str(torch.cuda.get_device_name(0)),
        "compute_capability": f"{getattr(p, 'major', '?')}.{getattr(p, 'minor', '?')}",
        "multi_processor_count": int(p.multi_processor_count),
        "max_threads_per_multi_processor": (int(p.max_threads_per_multi_processor)
                                            if hasattr(p, "max_threads_per_multi_processor") else None),
        "warp_size": int(p.warp_size) if hasattr(p, "warp_size") else None,
        "is_integrated": bool(p.is_integrated) if hasattr(p, "is_integrated") else None,
        "total_memory_bytes": int(p.total_memory),
        "how_obtained": "torch.cuda.get_device_properties(0); max threads per SM is read from the "
                        "max_threads_per_multi_processor attribute when this torch build exposes it, otherwise it is "
                        "inferred in stage C as T_found / (256 * multi_processor_count)",
        "warnings": [str(w.message)[:300] for w in caught_warnings],
    })
    mt = out["max_threads_per_multi_processor"]
    if mt:
        out["predicted_full_grid_threads"] = out["multi_processor_count"] * (mt // 256) * 256
    out["formula"] = FORMULA
    out["formula_source"] = PYTORCH_SOURCE
    out["status"] = "PASS"
    say(f"  torch {out['torch']} (CUDA build {out['torch_cuda_build']}), {out['device_name']}, compute capability "
        f"{out['compute_capability']}, {out['multi_processor_count']} SMs, max threads per SM: {mt if mt else 'not exposed by this torch'}")
    if mt:
        say(f"  PyTorch formula predicts T = {out['multi_processor_count']} x ({mt} // 256) x 256 = {out['predicted_full_grid_threads']:,}")
    for w in out["warnings"]:
        say(f"  torch warning: {w}")
    return out


# ----------------------------------------------------------------------------------------------
# Stage C - mechanism test on the device
# ----------------------------------------------------------------------------------------------
SEED_MAIN = 20261113
N_MAIN = 4_000_000
SWEEP_FLOOR = 700_000        # sample positions lie beyond the largest swept T (2600 * 256 = 665,600)
G_MAX = 2600
SWEEP_SAMPLES = 16384        # chance |r| for a wrong T is about 1/sqrt(16384) = 0.008; the largest of 2600 is about 0.03


def stage_c(dev, facts, skip_large=False):
    out = {"seed": SEED_MAIN, "numel": N_MAIN, "formula": FORMULA, "formula_source": PYTORCH_SOURCE, "checks": {}}
    ok = True
    rng = np.random.default_rng(0)
    pos = np.unique(rng.integers(SWEEP_FLOOR, N_MAIN, size=SWEEP_SAMPLES)).astype(np.int64)

    g_bits = dev.fill(N_MAIN, "float32", SEED_MAIN, 1.0)
    g = g_bits.view(F32)
    sw = sweep_layout(pos, g[pos], SEED_MAIN, 0, 1, G_MAX)
    (r1, g1), (r2, g2) = sw[0], sw[1]
    T = 256 * g1
    unique = bool(r1 > 0.9999 and abs(r2) < 0.1)
    ok &= unique
    out["T_found"] = T
    out["checks"]["sweep"] = {"grid_sizes_swept": [1, G_MAX], "sample_positions": int(pos.size), "best_T": T, "best_grid_blocks": g1,
                              "best_r": r1, "runner_up_T": 256 * g2, "runner_up_r": r2, "pass": unique}
    say(f"  sweep of T = 256*g, g = 1..{G_MAX}, on {pos.size} values beyond element {SWEEP_FLOOR:,}: best T = {T:,} "
        f"(g = {g1}) r = {r1:.6f}; runner-up T = {256 * g2:,} r = {r2:+.4f} -> {'unique' if unique else 'NOT UNIQUE'}")

    sms = facts.get("multi_processor_count")
    mt = facts.get("max_threads_per_multi_processor")
    pred = {"multi_processor_count": sms, "max_threads_per_multi_processor_reported": mt}
    if sms:
        bps = g1 / sms
        pred["blocks_per_sm_implied"] = bps
        integral = abs(bps - round(bps)) < 1e-9
        if mt:
            pred["T_predicted"] = sms * (mt // 256) * 256
            pred["pass"] = bool(pred["T_predicted"] == T)
        else:
            pred["max_threads_per_sm_inferred_range"] = [int(round(bps)) * 256, int(round(bps)) * 256 + 255] if integral else None
            pred["pass"] = bool(integral)
        say(f"  formula check: T / 256 / SMs = {g1} / {sms} = {bps:g} blocks per SM"
            + (f"; predicted T = {pred['T_predicted']:,}" if mt else f"; max threads per SM inferred in {pred['max_threads_per_sm_inferred_range']}")
            + f" -> {'ok' if pred['pass'] else 'MISMATCH'}")
    else:
        pred["pass"] = None
        say("  formula check: SM count unknown (simulated device); skipped")
    if pred["pass"] is not None:
        ok &= pred["pass"]
    out["checks"]["formula"] = pred

    # full-tensor comparison at float32 bit level, both contraction variants
    plan, _ = fill_plan(N_MAIN, 4, T, 0)
    best = None
    for fma in (True, False):
        e_bits = emulate_at(np.arange(N_MAIN, dtype=np.int64), plan, SEED_MAIN, fma).view(np.uint32)
        dist = ulp_distance(g_bits, e_bits)
        rec = {"fma_contracted_v": fma, "values": N_MAIN, "bit_identical": int((dist == 0).sum()), "within_1_ulp": int((dist <= 1).sum()),
               "within_2_ulp": int((dist <= 2).sum()), "fraction_bit_identical": float((dist == 0).mean()),
               "pearson_r": pearson(g, e_bits.view(F32))}
        if best is None or rec["bit_identical"] > best["bit_identical"]:
            best = rec
        out["checks"].setdefault("float32_bit_agreement", []).append(rec)
        say(f"  float32 bit agreement with correctly rounded emulator ({'fused' if fma else 'unfused'} v): {rec['bit_identical']:,}/{N_MAIN:,} identical "
            f"({100 * rec['fraction_bit_identical']:.4f}%), within 1 ulp {100 * rec['within_1_ulp'] / N_MAIN:.4f}%, r = {rec['pearson_r']:.9f}")
    fma = best["fma_contracted_v"]
    out["fma_variant_used"] = fma
    good = best["pearson_r"] > 0.999999
    ok &= good
    out["checks"]["float32_overall"] = {"pass": bool(good), "note": "pass criterion is r > 0.999999; bit-identical fraction is reported, not gated (stage D)"}

    # the three data-centre layouts must not match natively
    nat = []
    for label, Tn in (("132 SMs x 2048", 270336), ("148 SMs x 2048", 303104), ("152 SMs x 2048", 311296)):
        r = pearson(g[pos], normal_core(pos, Tn, SEED_MAIN, 0, fma)[0])
        same = Tn == T
        nat.append({"layout": label, "T": Tn, "r": r, "is_this_device": same, "pass": bool(same or abs(r) < 0.1)})
        ok &= nat[-1]["pass"]
        say(f"  native output vs {label} (T = {Tn:,}): r = {r:+.4f} -> {'this device has that layout' if same else ('no match' if abs(r) < 0.1 else 'UNEXPECTED MATCH')}")
    out["checks"]["native_vs_datacentre_layouts"] = nat

    # other dtypes: same seed, so the stored value must be the float32 value rounded to the dtype
    dt = []
    for dtype in ("bfloat16", "float16"):
        b = dev.fill(N_MAIN, dtype, SEED_MAIN, 1.0)
        from_gpu = to_bits(g, dtype)
        from_emu = to_bits(emulate_at(pos, plan, SEED_MAIN, fma), dtype)
        rec = {"dtype": dtype, "identical_to_own_float32_fill_rounded": float((b == from_gpu).mean()),
               "identical_to_emulator_at_samples": float((b[pos] == from_emu).mean())}
        rec["pass"] = bool(rec["identical_to_own_float32_fill_rounded"] > 0.99999 and rec["identical_to_emulator_at_samples"] > 0.99)
        ok &= rec["pass"]
        dt.append(rec)
        say(f"  {dtype}: identical to this device's float32 fill rounded to {dtype}: {100 * rec['identical_to_own_float32_fill_rounded']:.5f}%; "
            f"identical to emulator at samples: {100 * rec['identical_to_emulator_at_samples']:.3f}% -> {'ok' if rec['pass'] else 'UNEXPECTED'}")
    out["checks"]["other_dtypes"] = dt

    # std scaling
    b = dev.fill(N_MAIN, "float32", SEED_MAIN, 0.006)
    frac = float((b == scale_f32(g, 0.006).view(np.uint32)).mean())
    rec = {"std": 0.006, "identical_to_float32(std1_value * float32(std))": frac, "pass": bool(frac > 0.99999)}
    ok &= rec["pass"]
    out["checks"]["std_scaling"] = rec
    say(f"  normal_(0, 0.006) equals float32(z * float32(0.006)) for {100 * frac:.5f}% of values -> {'ok' if rec['pass'] else 'UNEXPECTED'}")

    # 2-D tensor: row-major linear index
    shape = (977, 4096)
    n2 = shape[0] * shape[1]
    pos2 = np.unique(rng.integers(SWEEP_FLOOR, n2, size=4096)).astype(np.int64)
    b = dev.fill(n2, "float32", SEED_MAIN + 1, 1.0, shape=shape)
    r = pearson(b.view(F32)[pos2], emulate_at(pos2, fill_plan(n2, 4, T, 0)[0], SEED_MAIN + 1, fma))
    rec = {"shape": list(shape), "r_row_major": r, "pass": bool(r > 0.999999)}
    ok &= rec["pass"]
    out["checks"]["two_dimensional"] = rec
    say(f"  2-D tensor {shape}: r = {r:.9f} with index = row * width + column -> {'ok' if rec['pass'] else 'UNEXPECTED'}")

    # small tensors: fewer blocks than the device grid, so no device fingerprint
    small = []
    for n in (10_000, T, T + 1000):
        b = dev.fill(n, "float32", SEED_MAIN + 2, 1.0)
        pl = fill_plan(n, 4, T, 0)[0]
        r = pearson(b.view(F32), emulate_at(np.arange(n, dtype=np.int64), pl, SEED_MAIN + 2, fma))
        small.append({"numel": n, "T_effective": pl[0][3], "r": r, "pass": bool(r > 0.999999)})
        ok &= small[-1]["pass"]
        say(f"  small tensor numel = {n:,}: T_effective = {pl[0][3]:,}, r = {r:.9f} -> {'ok' if small[-1]['pass'] else 'UNEXPECTED'}")
    out["checks"]["small_tensors"] = small

    # two fills in a row: the second starts at the offset the first one consumed
    n_a, n_b = N_MAIN, 1_000_000
    dev.fill(n_a, "float32", SEED_MAIN + 3, 1.0)
    b = dev.fill(n_b, "float32", SEED_MAIN + 3, 1.0, reseed=False)
    off = offset_inc(n_a, eff_threads(n_a, T))
    pos3 = np.unique(rng.integers(0, n_b, size=4096)).astype(np.int64)
    r_ok = pearson(b.view(F32)[pos3], emulate_at(pos3, fill_plan(n_b, 4, T, off)[0], SEED_MAIN + 3, fma))
    r_0 = pearson(b.view(F32)[pos3], emulate_at(pos3, fill_plan(n_b, 4, T, 0)[0], SEED_MAIN + 3, fma))
    rec = {"first_numel": n_a, "second_numel": n_b, "predicted_offset_of_second_fill": off, "r_at_predicted_offset": r_ok,
           "r_at_offset_0": r_0, "pass": bool(r_ok > 0.999999 and abs(r_0) < 0.1)}
    ok &= rec["pass"]
    out["checks"]["offset_chaining"] = rec
    say(f"  second fill without reseeding: predicted Philox offset {off}: r = {r_ok:.9f}; offset 0: r = {r_0:+.4f} -> {'ok' if rec['pass'] else 'UNEXPECTED'}")
    del g_bits, g, b

    out["status"] = "PASS" if ok else "FAIL"

    # large tensor (> 2**31 bytes): reviewer's derivation of the split-fill offsets (explains Kimi-K3's offset 3776)
    lt = {"status": "SKIPPED"}
    try:
        N = 1_100_000_000
        free = dev.free_bytes()
        if skip_large:
            lt["reason"] = "--skip-large"
        elif free is not None and free < 8 * 2 ** 30:
            lt["reason"] = f"less than 8 GiB free on the device ({free} bytes)"
        else:
            plan_l, _ = fill_plan(N, 2, T, 0)
            half = N // 2
            p_lo = np.unique(rng.integers(SWEEP_FLOOR, half, size=2048)).astype(np.int64)
            p_hi = np.unique(rng.integers(half + SWEEP_FLOOR, N, size=2048)).astype(np.int64)
            allp = np.concatenate([p_lo, p_hi])
            vals = bits_to_f64(dev.fill(N, "bfloat16", SEED_MAIN + 4, 1.0, positions=allp), "bfloat16")
            v_lo, v_hi = vals[:p_lo.size], vals[p_lo.size:]
            nosplit = [(0, N, 0, eff_threads(N, T))]
            lt = {"numel": N, "dtype": "bfloat16", "bytes": 2 * N, "predicted_segments": [list(map(int, s)) for s in plan_l],
                  "r_first_half_split_model": pearson(v_lo, emulate_at(p_lo, plan_l, SEED_MAIN + 4, fma)),
                  "r_second_half_split_model": pearson(v_hi, emulate_at(p_hi, plan_l, SEED_MAIN + 4, fma)),
                  "r_first_half_no_split_model": pearson(v_lo, emulate_at(p_lo, nosplit, SEED_MAIN + 4, fma)),
                  "r_second_half_no_split_model": pearson(v_hi, emulate_at(p_hi, nosplit, SEED_MAIN + 4, fma))}
            good = (lt["r_first_half_split_model"] > 0.9999 and lt["r_second_half_split_model"] > 0.9999
                    and abs(lt["r_first_half_no_split_model"]) < 0.1 and abs(lt["r_second_half_no_split_model"]) < 0.1)
            lt["status"] = "PASS" if good else "FAIL"
            say(f"  large tensor ({N:,} bfloat16 = {2 * N / 2 ** 30:.2f} GiB): split-fill model r = {lt['r_first_half_split_model']:.6f} / "
                f"{lt['r_second_half_split_model']:.6f} (first / second half), no-split model r = {lt['r_first_half_no_split_model']:+.4f} / "
                f"{lt['r_second_half_no_split_model']:+.4f}; predicted offsets {[s[2] for s in plan_l]}")
    except Exception as e:  # informative only; never fails stage C
        lt = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
    out["large_tensor_split"] = lt
    verdict_line("C+", lt["status"], "large-tensor split-fill offsets (reviewer's derivation; informative, not part of stage C's verdict)"
                 + (f" - {lt.get('reason') or lt.get('error')}" if lt["status"] in ("SKIPPED", "ERROR") else ""))
    return out


# ----------------------------------------------------------------------------------------------
# Stage D - is NVIDIA device math correctly rounded? does this GPU reproduce the public rows?
# ----------------------------------------------------------------------------------------------
def stage_d(dev, T, fma, skip_direct=False):
    out = {"T_device": T, "reference_public_weights": REFERENCE_FLIP_RATES}
    # D1: generic deviation statistics on this device's own output
    g_bits = dev.fill(N_MAIN, "float32", SEED_MAIN, 1.0)
    plan, _ = fill_plan(N_MAIN, 4, T, 0)
    hist = {"0": 0, "1": 0, "2": 0, "3-16": 0, "17-256": 0, ">256": 0}
    cls = {"near_zero": {"n": 0, "identical": 0, "gt16": 0, "flips": 0}, "other": {"n": 0, "identical": 0, "gt16": 0, "flips": 0}}
    for a in range(0, N_MAIN, 1 << 20):
        li = np.arange(a, min(a + (1 << 20), N_MAIN), dtype=np.int64)
        z, aux = normal_core(li, plan[0][3], SEED_MAIN, 0, fma, want_aux=True)
        gb = g_bits[a:a + li.size]
        dist = ulp_distance(gb, z.view(np.uint32))
        for key, m in (("0", dist == 0), ("1", dist == 1), ("2", dist == 2), ("3-16", (dist >= 3) & (dist <= 16)),
                       ("17-256", (dist >= 17) & (dist <= 256)), (">256", dist > 256)):
            hist[key] += int(m.sum())
        near = np.abs(aux["trig"]) < 0.02
        flips = f32_to_bf16_bits(scale_f32(gb.view(F32), 0.006)) != f32_to_bf16_bits(scale_f32(z, 0.006))
        for name, m in (("near_zero", near), ("other", ~near)):
            cls[name]["n"] += int(m.sum())
            cls[name]["identical"] += int((dist[m] == 0).sum())
            cls[name]["gt16"] += int((dist[m] > 16).sum())
            cls[name]["flips"] += int(flips[m].sum())
    total_flips = cls["near_zero"]["flips"] + cls["other"]["flips"]
    rate = total_flips / N_MAIN
    share = cls["near_zero"]["flips"] / total_flips if total_flips else None
    base = cls["near_zero"]["n"] / N_MAIN
    if rate < 2e-5:
        v1 = "NO_DEVIATION"       # device math behaves like correctly rounded math at this resolution
    elif 1.5e-4 <= rate <= 4.0e-4 and share is not None and share >= 0.4:
        v1 = "CONSISTENT_WITH_PUBLIC_WEIGHTS"
    else:
        v1 = "DIFFERENT_FROM_PUBLIC_WEIGHTS"
    out["D1_generic"] = {
        "values": N_MAIN, "float32_ulp_distance_histogram": hist, "fraction_float32_bit_identical": hist["0"] / N_MAIN,
        "near_zero_definition": "|sin v| < 0.02 (even lanes) or |cos v| < 0.02 (odd lanes)", "near_zero_fraction_of_values": base,
        "by_class": cls, "bf16_flips_at_std_0.006": total_flips, "bf16_flip_rate": rate, "near_zero_share_of_flips": share,
        "enrichment_of_flips_near_zero": (share / base) if (share is not None and base > 0) else None, "verdict": v1}
    say(f"  D1 float32 ulp distance, GPU vs correctly rounded: " + ", ".join(f"{k}: {v:,}" for k, v in hist.items()))
    say(f"  D1 bf16 values (x 0.006) that differ: {total_flips:,}/{N_MAIN:,} = {rate:.3e} per value; share near sin/cos zeros: "
        f"{'n/a' if share is None else f'{100 * share:.1f}%'} (such values are {100 * base:.2f}% of all) -> {v1}")
    say("     reference, public weights vs correctly rounded math: V4-Flash 2.45e-4 (60% near zeros), V4-Pro 2.51e-4 (61%), V3 2.63e-4 (58%)")
    del g_bits

    # D2: reproduce the public DeepSeek-V4-Flash rows with stock PyTorch on this device.
    # Element li of a T_b-thread fill is thread t = li % T_b at step q = li // T_b. If t < T_device, stock PyTorch on
    # this device computes exactly that Philox call at position q * T_device + t of a large enough tensor.
    d2 = {"status": "SKIPPED"}
    try:
        prov, rows, pub = load_sample()
        d = pub.shape[1]
        li = (rows[:, None] * d + np.arange(d, dtype=np.int64)[None, :]).ravel()
        pubf = pub.ravel()
        Tb = V4F["T"]
        t, q = li % Tb, li // Tb
        cov = t < T
        N = int((q[cov].max() + 1) * T) if cov.any() else 0
        free = dev.free_bytes()
        if skip_direct:
            d2["reason"] = "--skip-direct"
        elif not cov.any():
            d2["reason"] = "no sample value has a thread index below this device's thread count"
        elif not can_use_32bit(N, 2):
            d2["reason"] = f"needed tensor ({N:,} bfloat16) exceeds 2**31 bytes, which would change the offsets"
        elif free is not None and free < 3 * N * 2:
            d2["reason"] = f"not enough free device memory for {N:,} bfloat16 values"
        else:
            p = (q[cov] * T + t[cov]).astype(np.int64)
            gpu = dev.fill(N, "bfloat16", V4F["seed"], V4F["std"], positions=p)
            emu = f32_to_bf16_bits(scale_f32(normal_core(li[cov], Tb, V4F["seed"], 0, fma)[0], V4F["std"]))
            pubc = pubf[cov]
            disc = emu != pubc           # values where correctly rounded math disagrees with the public weights
            d2 = {"sample": f"{prov['source']} @ {prov['revision'][:10]} {prov['tensor']} rows {prov['row_ranges_inclusive']}",
                  "layout_of_public_rows": Tb, "seed": V4F["seed"], "std": V4F["std"],
                  "tensor_generated_on_device": {"numel": N, "dtype": "bfloat16"},
                  "values_reachable_on_this_device": int(cov.sum()), "values_in_sample": int(cov.size),
                  "gpu_identical_to_public": int((gpu == pubc).sum()),
                  "correctly_rounded_identical_to_public": int((emu == pubc).sum()),
                  "discriminating_values": int(disc.sum()),
                  "discriminating_gpu_equals_public": int((gpu[disc] == pubc[disc]).sum()),
                  "discriminating_gpu_equals_correctly_rounded": int((gpu[disc] == emu[disc]).sum())}
            d2["fraction_gpu_identical_to_public"] = d2["gpu_identical_to_public"] / d2["values_reachable_on_this_device"]
            d2["status"] = "PASS" if d2["gpu_identical_to_public"] == d2["values_reachable_on_this_device"] else "FAIL"
            say(f"  D2 stock PyTorch on this device, read out in T = {Tb:,} order: {d2['gpu_identical_to_public']:,}/{d2['values_reachable_on_this_device']:,} "
                f"public bf16 values reproduced bit for bit (correctly rounded math: {d2['correctly_rounded_identical_to_public']:,})")
            say(f"  D2 of the {d2['discriminating_values']} values where correctly rounded math disagrees with the public weights, this GPU agrees "
                f"with the public weights in {d2['discriminating_gpu_equals_public']} and with correctly rounded math in {d2['discriminating_gpu_equals_correctly_rounded']}")
    except Exception as e:
        d2 = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
    out["D2_public_rows"] = d2
    if d2["status"] in ("PASS", "FAIL"):
        out["status"] = d2["status"]
        out["basis"] = "D2 (bit-for-bit reproduction of public rows by this GPU)"
    else:
        out["status"] = "PASS" if v1 == "CONSISTENT_WITH_PUBLIC_WEIGHTS" else "FAIL"
        out["basis"] = f"D1 only (D2 {d2['status'].lower()}: {d2.get('reason') or d2.get('error')})"
    return out


# ----------------------------------------------------------------------------------------------
# Stage E - spoofing demo
# ----------------------------------------------------------------------------------------------
def stage_e(torch_mod, facts, T_native):
    out = {"label": "DEMONSTRATION ONLY. It shows that the layout is whatever the software writes. It does not show that "
                    "anyone did this, and it says nothing about which hardware any published model was made on.",
           "demo_seed": 424242, "shape": [512, 4096], "std": 0.02, "native_T_of_this_device": T_native,
           "device": facts.get("device_name"), "device_sm_count": facts.get("multi_processor_count"), "runs": []}
    ok = True
    n = 512 * 4096
    rng = np.random.default_rng(1)
    pos = np.unique(rng.integers(SWEEP_FLOOR, n, size=SWEEP_SAMPLES)).astype(np.int64)
    for T_spoof in (303104, 311296):
        x = scale_f32(emulate_at(np.arange(n, dtype=np.int64), [(0, n, 0, T_spoof)], 424242), 0.02)
        where = "host memory only"
        if torch_mod is not None and torch_mod.cuda.is_available():
            t = torch_mod.from_numpy(x.copy()).to("cuda")       # the tensor now lives on this GPU
            x = t.cpu().numpy()
            del t
            where = f"stored on and read back from {facts.get('device_name')} ({facts.get('multi_processor_count')} SMs)"
        sw = sweep_layout(pos, x[pos], 424242, 0, 1, G_MAX)
        (r1, g1), (r2, g2) = sw[0], sw[1]
        good = bool(256 * g1 == T_spoof and r1 > 0.9999 and abs(r2) < 0.1)
        ok &= good
        out["runs"].append({"layout_written": T_spoof, "layout_read_back": 256 * g1, "r": r1, "runner_up_T": 256 * g2, "runner_up_r": r2,
                            "tensor": where, "pass": good})
        say(f"  wrote layout T = {T_spoof:,} in software ({where}); the layout reader reports T = {256 * g1:,} "
            f"(= {g1 / 8:g} SMs x 2048), r = {r1:.6f}; runner-up r = {r2:+.4f}"
            + (f"; this device's native T is {T_native:,}" if T_native else ""))
    out["status"] = "PASS" if ok else "FAIL"
    return out


# ----------------------------------------------------------------------------------------------
# Stage F (optional, --blind-search) - blind seed recovery
# ----------------------------------------------------------------------------------------------
# Input: only the bundled rows, their row indices and the tensor's row width (public file metadata).
# Not given: the seed, the thread count, the init scale. Searched: every seed in --seed-range (default all
# 2**32) under each candidate thread count, generator offset 0 only.
#   1. scale: sigma = rms of the bundled values (an untouched N(0, std) row has rms = std).
#   2. probe: the 32 largest-magnitude values of the first PROBE_ROWS bundled rows. For every seed, compute the
#      one Philox call that would produce probe value 1 and keep the seed only if it lands within TOL of it;
#      survivors are tested on probe values 2, 3, ... in turn.
#   3. confirmation: a seed that passes all 32 probe values is checked on the remaining rows, which played no
#      part in the search (held-out), by correlation.
# The arithmetic is plain int64 / float32 tensor code (torch on the GPU if available, else torch CPU, else numpy).
F_K = 32
F_PROBE_ROWS = 4
F_TOL = 0.02
F_KEEP = 20                                  # how many of the best non-winning candidates to remember
F_DEFAULT_SMS = (132, 148, 152, 108, 144)   # 108 = A100-class layout (real, but wrong here); 144 = made-up decoy
F_CHECKPOINT = "blind_search_checkpoint.json"
_P_M0, _P_M1, _P_W0, _P_W1, _P_MASK = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85, 0xFFFFFFFF


def philox_generic(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10 on Python ints and/or int64 arrays (torch or numpy), mixed freely.
    For int64 arrays the 64-bit product wraps; (p >> 32) & mask and p & mask still give its true high and low words."""
    for _ in range(10):
        p0 = c0 * _P_M0
        p1 = c2 * _P_M1
        n0 = ((p1 >> 32) & _P_MASK) ^ c1 ^ k0
        n2 = ((p0 >> 32) & _P_MASK) ^ c3 ^ k1
        c0, c1, c2, c3 = n0, p1 & _P_MASK, n2, p0 & _P_MASK
        k0 = (k0 + _P_W0) & _P_MASK
        k1 = (k1 + _P_W1) & _P_MASK
    return c0, c1, c2, c3


class _NumpyBackend:
    name, device = "numpy", "cpu"

    def arange(self, a, b):
        return np.arange(a, b, dtype=np.int64)

    def from_ints(self, vals):
        return np.array(vals, dtype=np.int64)

    def f32(self, x):
        return x.astype(F32)

    sin, cos, log, sqrt, absolute = staticmethod(np.sin), staticmethod(np.cos), staticmethod(np.log), staticmethod(np.sqrt), staticmethod(np.abs)

    def nonzero(self, m):
        return np.nonzero(m)[0]

    def numpy(self, x):
        return np.asarray(x)


class _TorchBackend:
    name = "torch"

    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.sin, self.cos, self.log, self.sqrt, self.absolute = torch.sin, torch.cos, torch.log, torch.sqrt, torch.abs

    def arange(self, a, b):
        return self.torch.arange(a, b, dtype=self.torch.int64, device=self.device)

    def from_ints(self, vals):
        return self.torch.tensor(vals, dtype=self.torch.int64, device=self.device)

    def f32(self, x):
        return x.to(self.torch.float32)

    def nonzero(self, m):
        return self.torch.nonzero(m).flatten()

    def numpy(self, x):
        return x.cpu().numpy()


def _probe_z(be, pr, seeds):
    """Standard-normal value that seed(s) would give at one probe position (float32, fast math; prefilter only)."""
    w = philox_generic(pr["c0"], pr["c1"], pr["c2"], pr["c3"], seeds, 0)
    x, y = (w[0], w[1]) if pr["lane"] < 2 else (w[2], w[3])
    u = be.f32(x) * (2.0 ** -32) + (2.0 ** -33)
    v = be.f32(y) * float(_INV_2PI32) + float(_HALF_INV_2PI32)
    s = be.sqrt(be.log(u) * -2.0)
    return (be.sin(v) if pr["lane"] % 2 == 0 else be.cos(v)) * s


def _fmt_eta(sec):
    if sec is None or not math.isfinite(sec):
        return "?"
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def stage_f(args, torch_mod, facts):
    prov, rows, bits = load_sample()
    d = bits.shape[1]
    W = bf16_bits_to_f32(bits).astype(F64)
    sigma = float(np.sqrt((W ** 2).mean()))
    probe_rows, held_rows = np.arange(F_PROBE_ROWS), np.arange(F_PROBE_ROWS, len(rows))
    flat = np.abs(W[probe_rows]).ravel()
    top = np.argsort(-flat)[:F_K]
    probe_li = (rows[probe_rows][top // d] * d + top % d).astype(np.int64)
    probe_w = W[probe_rows].ravel()[top]
    held_li = (rows[held_rows][:, None] * d + np.arange(d, dtype=np.int64)[None, :]).ravel()
    held_w = W[held_rows].ravel()

    lo, hi = (int(x) for x in args.seed_range.split(":"))
    if not (0 <= lo < hi <= 2 ** 32):
        raise ValueError("--seed-range must satisfy 0 <= lo < hi <= 4294967296")
    native_T = None
    if facts.get("multi_processor_count") and facts.get("max_threads_per_multi_processor"):
        native_T = facts["multi_processor_count"] * (facts["max_threads_per_multi_processor"] // 256) * 256
    if args.layouts:
        layouts = [int(x) for x in args.layouts.split(",")]
    else:
        layouts = [s * 2048 for s in F_DEFAULT_SMS] + ([native_T] if native_T and native_T not in [s * 2048 for s in F_DEFAULT_SMS] else [])
    if any(T <= 0 or T % 256 for T in layouts):
        raise ValueError("every layout must be a positive multiple of 256")

    use_cuda = bool(torch_mod is not None and torch_mod.cuda.is_available() and args.device in ("auto", "cuda"))
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if torch_mod is not None:
        be = _TorchBackend(torch_mod, "cuda" if use_cuda else "cpu")
    else:
        be = _NumpyBackend()
    batch = args.batch or (1 << 24 if use_cuda else 1 << 20)

    # self-check of the search arithmetic on this backend before trusting hours of it
    for c, k, want in KAT:
        got_int = philox_generic(*c, *k)
        got_arr = [int(be.numpy(x)[0]) for x in philox_generic(*[be.from_ints([v]) for v in c], be.from_ints([k[0]]), be.from_ints([k[1]]))]
        if tuple(got_int) != want or tuple(got_arr) != want:
            raise RuntimeError(f"Philox self-check failed on backend {be.name}/{be.device}; do not use this backend")
    z_chk = float(be.numpy(_probe_z(be, {"c0": 3, "c1": 0, "c2": 77, "c3": 0, "lane": 2}, be.from_ints([424242])))[0])
    z_ref = float(normal_core(np.array([77 + 14 * 1000]), 1000, 424242, 0)[0][0])   # thread 77, step 3, lane 2 of a T=1000 fill
    if abs(z_chk - z_ref) > 1e-4 * max(1.0, abs(z_ref)):
        raise RuntimeError(f"Box-Muller self-check failed on backend {be.name}/{be.device}: {z_chk} vs {z_ref}")

    probes = {}
    for T in layouts:
        t, q = probe_li % T, probe_li // T
        probes[T] = [{"c0": int(q[i] // 4) & _P_MASK, "c1": int(q[i] // 4) >> 32, "c2": int(t[i]) & _P_MASK, "c3": int(t[i]) >> 32,
                      "lane": int(q[i] % 4), "w": float(probe_w[i])} for i in range(F_K)]

    config = {"sample_sha256": prov["sha256"], "layouts": layouts, "seed_range": [lo, hi], "K": F_K, "probe_rows": F_PROBE_ROWS,
              "tol": F_TOL, "kit_version": KIT_VERSION, "state_format": 2}
    ck_path = os.path.join(KIT_DIR, F_CHECKPOINT)
    state = {"config": config, "next_seed": lo, "elapsed": 0.0, "others": [], "passed_first_probe_value": 0, "complete": []}
    if os.path.exists(ck_path) and not args.restart:
        with open(ck_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        if old.get("config") == config:
            state = old
            say(f"  resuming from checkpoint: next seed {state['next_seed']:,} ({100 * (state['next_seed'] - lo) / (hi - lo):.1f}% done, "
                f"{_fmt_eta(state['elapsed'])} already spent)")
        else:
            raise RuntimeError(f"{F_CHECKPOINT} belongs to a different search (other range, layouts or kit version). "
                               "Re-run with --restart to discard it.")

    def save():
        tmp = ck_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, ck_path)

    say(f"  input: {len(rows)} bundled rows x {d}; scale estimated from the data: sigma = {sigma:.6g}")
    say(f"  probe: {F_K} largest-magnitude values of the first {F_PROBE_ROWS} rows; held-out for confirmation: the other {len(held_rows)} rows ({held_w.size:,} values)")
    say(f"  candidate thread counts: {', '.join(f'{T:,}' for T in layouts)}" + (f" (native T of this device: {native_T:,})" if native_T else ""))
    say(f"  seeds {lo:,} .. {hi - 1:,} ({hi - lo:,} seeds x {len(layouts)} layouts), generator offset 0 only; backend {be.name} on {be.device}, batch {batch:,}")

    t_run = time.time()
    last_print = last_save = t_run
    done_at_start = state["next_seed"]
    interrupted = False
    clock = {"t": t_run}

    def tick():   # add the time since the last tick to the total stored in the checkpoint
        now_ = time.time()
        state["elapsed"] += now_ - clock["t"]
        clock["t"] = now_

    try:
        s0 = state["next_seed"]
        while s0 < hi:
            s1 = min(s0 + batch, hi)
            seeds = be.arange(s0, s1)
            b_pass1, b_others, b_complete = 0, [], []   # committed only when the whole batch is done (safe to interrupt)
            floor = state["others"][-1]["probe_values_passed"] if len(state["others"]) >= F_KEEP else 0
            for T in layouts:
                pr = probes[T]
                surv = seeds
                for k in range(F_K):
                    z = _probe_z(be, pr[k], surv)
                    ok = be.absolute(z * sigma - pr[k]["w"]) <= F_TOL * abs(pr[k]["w"])
                    if k >= 1:   # the seeds dropped here matched probe values 1..k but not k+1
                        if k == 1:
                            b_pass1 += int(surv.shape[0])
                        if k > floor:
                            b_others += [{"seed": int(s), "T": T, "probe_values_passed": k} for s in be.numpy(surv[~ok]).tolist()]
                    surv = surv[be.nonzero(ok)]
                    if surv.shape[0] == 0:
                        break
                else:
                    b_complete += [{"seed": int(s), "T": T, "probe_values_passed": F_K} for s in be.numpy(surv).tolist()]
            state["passed_first_probe_value"] += b_pass1
            state["others"] = sorted(state["others"] + b_others, key=lambda c: -c["probe_values_passed"])[:F_KEEP]
            for c in b_complete:
                state["complete"].append(c)
                say(f"  >>> seed {c['seed']:,} at T = {c['T']:,} matches all {F_K} probe values (confirmation on held-out rows follows at the end)")
            s0 = s1
            state["next_seed"] = s0
            now = time.time()
            if now - last_print >= 10 or s0 >= hi:
                rate = (s0 - done_at_start) * len(layouts) / max(now - t_run, 1e-9)
                eta = (hi - s0) * len(layouts) / rate if rate > 0 else None
                say(f"  progress {100 * (s0 - lo) / (hi - lo):6.2f}%  next seed {s0:,}  {rate / 1e6:8.2f} M seed-layouts/s  ETA {_fmt_eta(eta)}  "
                    f"matched 1st probe value: {state['passed_first_probe_value']:,}  matched all {F_K}: {len(state['complete'])}")
                last_print = now
            if now - last_save >= 30 or s0 >= hi:
                tick()
                last_save = now
                save()
    except KeyboardInterrupt:
        interrupted = True
        tick()
        save()
        say(f"  interrupted; progress saved to {F_CHECKPOINT} (next seed {state['next_seed']:,}). Run the same command again to resume.")
    run_seconds = time.time() - t_run
    searched_now = state["next_seed"] - done_at_start
    rate = searched_now * len(layouts) / max(run_seconds, 1e-9)

    # confirmation on held-out rows with the reference emulator (numpy, correctly rounded)
    def score(seed, T):
        zp = normal_core(probe_li, T, seed, 0)[0].astype(F64)
        cos = float(np.dot(probe_w, zp) / math.sqrt(float(np.dot(probe_w, probe_w)) * float(np.dot(zp, zp))))
        return {"heldout_r": pearson(held_w, normal_core(held_li, T, seed, 0)[0]), "probe_cosine": cos}
    confirmed, rejected = [], []
    for c in state["complete"]:
        rec = dict(c, **score(c["seed"], c["T"]))
        (confirmed if rec["heldout_r"] > 0.99 else rejected).append(rec)
    near = [dict(c, **score(c["seed"], c["T"])) for c in state["others"]] + rejected
    runner = max(near, key=lambda c: (c["probe_values_passed"], abs(c["heldout_r"])), default=None)

    full = (lo, hi) == (0, 2 ** 32)
    finished = state["next_seed"] >= hi and not interrupted
    out = {"seed_range": [lo, hi], "full_32_bit_range": full, "finished": finished, "next_seed": state["next_seed"],
           "layouts": layouts, "native_T_of_this_device": native_T, "offsets_searched": [0],
           "sigma_estimated_from_data": sigma, "probe_values": F_K, "probe_rows": [int(r) for r in rows[probe_rows]],
           "heldout_rows": len(held_rows), "heldout_values": int(held_w.size), "tolerance": F_TOL,
           "backend": be.name, "device": be.device, "device_name": facts.get("device_name"), "batch": batch,
           "seconds_this_run": round(run_seconds, 2), "seconds_total": round(state["elapsed"], 2),
           "seed_layouts_per_second_this_run": rate, "confirmed": confirmed,
           "seed_layouts_matching_first_probe_value": state["passed_first_probe_value"],
           "note_on_first_probe_value": "wrong seeds match the first probe value by chance a few times per million; they then fail the next ones",
           "runner_up": runner, "best_other_candidates": near[:5], "checkpoint_file": F_CHECKPOINT,
           "caveat": "the search is told neither seed nor layout, but the bundled rows were chosen by the reviewer, who knew "
                     "which rows are untouched; and only generator offset 0 is searched"}
    if not finished:
        out["status"] = "INCOMPLETE"
        out["reason"] = "interrupted before the end of the seed range; run the same command again to resume"
    elif len(confirmed) == 1:
        out["status"] = "PASS"
        c = confirmed[0]
        out["recovered"] = {"seed": c["seed"], "T": c["T"], "sms_if_2048_threads_each": c["T"] / 2048, "heldout_r": c["heldout_r"]}
        out["equals_published_claim_seed_953049583_T_303104"] = bool(c["seed"] == V4F["seed"] and c["T"] == V4F["T"])
        say(f"  RECOVERED seed = {c['seed']:,}, T = {c['T']:,} (= {c['T'] / 2048:g} SMs x 2048): held-out r = {c['heldout_r']:.6f} on {held_w.size:,} values "
            f"not used by the search; equals the published claim: {out['equals_published_claim_seed_953049583_T_303104']}")
    elif len(confirmed) == 0:
        out["status"] = "FAIL" if full else "NOT_FOUND_IN_RANGE"
        out["reason"] = ("no seed in the full 32-bit range reproduces the held-out rows under any candidate layout" if full
                         else "no seed in the searched slice reproduces the held-out rows (expected unless the slice contains the true seed)")
        say(f"  no seed in {lo:,}..{hi - 1:,} reproduces the held-out rows under any candidate layout")
    else:
        out["status"] = "FAIL"
        out["reason"] = f"{len(confirmed)} different (seed, T) pairs reproduce the held-out rows; expected exactly one"
    if runner:
        say(f"  runner-up: seed {runner['seed']:,} at T = {runner['T']:,} passed {runner['probe_values_passed']} of {F_K} probe values, "
            f"probe cosine {runner['probe_cosine']:+.3f}, held-out r = {runner['heldout_r']:+.5f}")
    else:
        say(f"  runner-up: none (no other seed matched even the first of the {F_K} probe values)")
    say(f"  throughput this run: {rate / 1e6:.2f} million seed-layout evaluations per second on {be.name}/{be.device} "
        f"({searched_now:,} seeds x {len(layouts)} layouts in {run_seconds:.1f} s)")
    if rate > 0:
        say(f"  at this rate a full 2**32 sweep of {len(layouts)} layouts takes about {_fmt_eta(2 ** 32 * len(layouts) / rate)} (h:mm:ss)")
    return out


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="JSON file name inside this folder (default results.json; results_blind.json with --blind-search)")
    ap.add_argument("--blind-search", action="store_true",
                    help="run ONLY the optional stage F: recover seed and thread count from the bundled rows by brute force")
    ap.add_argument("--seed-range", default="0:4294967296", metavar="LO:HI", help="stage F: seeds LO <= s < HI (default: all 2**32)")
    ap.add_argument("--layouts", default="", metavar="T1,T2,...", help="stage F: thread counts to search (default: 132/148/152/108/144 x 2048 plus this device's own)")
    ap.add_argument("--batch", type=int, default=0, help="stage F: seeds per batch (default 2**24 on CUDA, 2**20 on CPU)")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"), help="stage F: where to run the search")
    ap.add_argument("--restart", action="store_true", help="stage F: discard an existing checkpoint instead of resuming")
    ap.add_argument("--stages", default="ABCDE", help="subset of stages to run, e.g. A or ABC")
    ap.add_argument("--skip-large", action="store_true", help="skip the 2.2 GiB large-tensor check in stage C")
    ap.add_argument("--skip-direct", action="store_true", help="skip D2 (needs about 0.3 GiB of device memory on a 48-SM GPU)")
    ap.add_argument("--simulate-device", type=int, default=0, metavar="T",
                    help="DEVELOPER SELF-TEST ONLY: replace the GPU by the emulator with T threads. Results are meaningless for hardware.")
    a = ap.parse_args(argv)
    main.argv_used = list(argv) if argv is not None else sys.argv[1:]

    out_path = os.path.abspath(os.path.join(KIT_DIR, a.out or ("results_blind.json" if a.blind_search else "results.json")))
    if os.path.commonpath([out_path, KIT_DIR]) != KIT_DIR:
        say("refusing to write outside the kit folder")
        return 2
    if a.blind_search:
        return main_blind(a, out_path)
    stages = set(a.stages.upper())
    results = {"kit_version": KIT_VERSION, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "argv": main.argv_used, "simulated_device": bool(a.simulate_device), "stages": {}}
    if a.simulate_device:
        say("=" * 100)
        say("SIMULATED DEVICE: stages C-E run against the emulator itself. THIS IS A CODE SELF-TEST, NOT A HARDWARE RESULT.")
        say("=" * 100)

    def run(name, title, fn, *args, **kw):
        say(f"\n=== Stage {name}: {title} ===")
        t0 = time.time()
        try:
            res = fn(*args, **kw)
        except Exception as e:
            res = {"status": "ERROR", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
        res["seconds"] = round(time.time() - t0, 2)
        results["stages"][name] = res
        extra = res.get("reason") or res.get("error") or res.get("basis") or ""
        verdict_line(name, res["status"], title + (f" - {extra}" if extra else ""))
        with open(out_path, "w", encoding="utf-8") as f:   # written after every stage, so a crash still leaves a file
            json.dump(results, f, indent=1)
        return res

    def skip(name, title, reason):
        results["stages"][name] = {"status": "SKIPPED", "reason": reason}
        say(f"\n=== Stage {name}: {title} ===")
        verdict_line(name, "SKIPPED", reason)

    if "A" in stages:
        run("A", "CPU self-test (no GPU involved)", stage_a)

    torch_mod, dev, facts, caught = None, None, {}, []
    no_gpu_reason = None
    if stages & set("BCDE"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                import torch as torch_mod  # noqa: F811
                if torch_mod.cuda.is_available():
                    torch_mod.cuda.init()
                    torch_mod.empty(1, device="cuda")
        except ImportError as e:
            torch_mod, no_gpu_reason = None, f"torch is not installed ({e})"
        except Exception as e:
            no_gpu_reason = f"CUDA could not be initialised: {type(e).__name__}: {e}"

    titles = {"B": "device facts", "C": "mechanism test with stock PyTorch on this GPU",
              "D": "NVIDIA device math vs correctly rounded math, and vs the public rows", "E": "spoofing demo (clearly labelled)"}
    if "B" in stages:
        if torch_mod is None:
            skip("B", titles["B"], no_gpu_reason or "torch is not installed")
        else:
            facts = run("B", titles["B"], stage_b, torch_mod, caught)
            if facts.get("status") == "SKIPPED":
                no_gpu_reason = facts.get("reason")
    if a.simulate_device:
        dev = SimulatedDevice(a.simulate_device)
        facts = dict(facts, device_name="SIMULATED", multi_processor_count=None)
    elif torch_mod is not None and no_gpu_reason is None and torch_mod.cuda.is_available():
        dev = TorchCudaDevice(torch_mod)
        if not facts.get("multi_processor_count"):   # stage B not requested: still need the facts
            try:
                facts = stage_b(torch_mod, caught)
            except Exception:
                facts = {}
    else:
        no_gpu_reason = no_gpu_reason or "torch.cuda.is_available() is False"

    c_res = None
    if "C" in stages:
        if dev is None:
            skip("C", titles["C"], f"no CUDA device: {no_gpu_reason}")
        else:
            c_res = run("C", titles["C"], stage_c, dev, facts, a.skip_large)
    T_native = (c_res or {}).get("T_found")
    if "D" in stages:
        if dev is None:
            skip("D", titles["D"], f"no CUDA device: {no_gpu_reason}")
        elif not T_native or (c_res or {}).get("checks", {}).get("sweep", {}).get("pass") is not True:
            skip("D", titles["D"], "needs the thread count found in stage C (run with C; C must find a unique T)")
        else:
            run("D", titles["D"], stage_d, dev, T_native, c_res.get("fma_variant_used", True), a.skip_direct)
    if "E" in stages:
        run("E", titles["E"], stage_e, None if a.simulate_device else (torch_mod if dev is not None else None), facts, T_native)
        if dev is None:
            results["stages"]["E"]["note"] = f"CPU only (no CUDA device: {no_gpu_reason}); the tensor never touched a GPU"
            say(f"  note: {results['stages']['E']['note']}")

    order = [s for s in "ABCDE" if s in results["stages"]]
    summary = {s: results["stages"][s]["status"] for s in order}
    results["summary"] = summary
    results["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    results["gpu_stages_ran"] = bool(dev is not None and not a.simulate_device)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    say("\n=== Summary ===")
    say("  " + "  ".join(f"{k}:{v}" for k, v in summary.items()) + ("   [SIMULATED DEVICE - not a hardware result]" if a.simulate_device else ""))
    if dev is None and stages & set("BCD"):
        say(f"  GPU stages did not run: {no_gpu_reason}. Only stage A (and the CPU part of E) says anything.")
    say(f"  results written to {os.path.relpath(out_path, KIT_DIR)}")
    bad = [k for k, v in summary.items() if v in ("FAIL", "ERROR")]
    return 1 if bad else 0


def main_blind(a, out_path):
    """Optional stage F on its own; writes its own JSON file."""
    results = {"kit_version": KIT_VERSION, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "argv": getattr(main, "argv_used", sys.argv[1:]), "stages": {}}
    say("\n=== Stage F (optional): blind recovery of seed and thread count from the bundled rows ===")
    torch_mod, facts = None, {}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import torch as torch_mod  # noqa: F811
            if a.device != "cpu" and torch_mod.cuda.is_available():
                facts = stage_b(torch_mod, caught)
    except ImportError:
        torch_mod = None
        say("  torch is not installed: falling back to numpy on the CPU (slow)")
    except Exception as e:
        say(f"  CUDA could not be used ({type(e).__name__}: {e}); continuing on the CPU")
        facts = {}
    t0 = time.time()
    try:
        res = stage_f(a, torch_mod, facts)
    except Exception as e:
        res = {"status": "ERROR", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
    res["seconds"] = round(time.time() - t0, 2)
    results["stages"]["F"] = res
    results["device_facts"] = facts
    results["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    verdict_line("F", res["status"], "blind seed recovery" + (f" - {res.get('reason') or res.get('error')}" if (res.get("reason") or res.get("error")) else ""))
    say(f"  results written to {os.path.relpath(out_path, KIT_DIR)}")
    return 1 if res["status"] in ("FAIL", "ERROR") else 0


if __name__ == "__main__":
    sys.exit(main())
