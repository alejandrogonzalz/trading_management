# RunPod Image Research — Flash-Attn + Unsloth + QLoRA

> **Why this document exists:** We hit a cascade of environment issues on the current
> pod that all trace back to starting from the wrong base image. This document captures
> the full debugging history and defines exactly what to look for in the next image.

---

## Current Pod (what went wrong)

| Field | Value |
|-------|-------|
| Template | `runpod-torch-v280` |
| GPU | A100 80GB PCIe (`sm_80`) |
| vCPU | 16 (AMD EPYC 7543, but `nproc` shows 128 hardware threads) |
| RAM | 125 GB |
| Container disk | 30 GB ← **too small** |
| Network volume | 100 GB at `/workspace` |
| Driver | 550.127.05 |
| `nvidia-smi` CUDA | 12.4 (host driver max) |
| Container CUDA toolkit | 12.8 (via RunPod forward compat) |
| System torch (image) | `2.8.0+cu128` |
| Our venv torch | `2.6.0+cu124` ← **wrong, installed by our script** |

---

## Root Cause Chain (full debugging history)

### Problem 1 — setup_runpod.sh detected the wrong CUDA version

`setup_runpod.sh` reads `nvidia-smi` to detect CUDA:

```bash
DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
# → returns "12.4" (host driver max, not container toolkit)
```

RunPod uses **CUDA Forward Compatibility** — the container has the full 12.8 toolkit
and can run cu128 code even though the host driver advertises 12.4. Our script didn't
know this and installed `torch==2.6.0+cu124` into the venv instead of the image's
intended `torch==2.8.0+cu128`.

**Fix needed in `setup_runpod.sh`:** detect toolkit from `nvcc --version`, not from
`nvidia-smi`:

```bash
# WRONG — reads host driver limit, not container toolkit
DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')

# CORRECT — reads what's actually installed and usable in this container
TOOLKIT_CUDA=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
```

### Problem 2 — flash-attn prebuilt wheel ABI mismatch

flash-attn 2.8+ changed its install mechanism: instead of compiling from source it
auto-downloads a prebuilt wheel from GitHub tagged `cxx11abiFALSE`. Those wheels were
compiled against **conda-distributed PyTorch** (CXX11 ABI = True). pip-installed
PyTorch uses the **old Itanium C++ ABI** (`std::string` mangles to `Ss`).

Confirmed by inspecting `libc10.so`:

```bash
nm -D .venv/lib/python3.12/site-packages/torch/lib/libc10.so | grep "c105Error.*SourceLocation"
# → _ZN3c105ErrorC2ENS_14SourceLocationESs   ← "Ss" = old ABI

# flash-attn prebuilt wheel needs:
# → _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
#   "NSt7__cxx11" = new CXX11 ABI  ← MISMATCH
```

**Symptom:**
```
ImportError: undefined symbol:
  _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

**Fix:** `FLASH_ATTENTION_FORCE_BUILD=TRUE` bypasses the prebuilt download and forces
actual source compilation against our torch headers (which use the correct ABI).

### Problem 3 — Source build compiled for 4 GPU architectures (slow)

Without `TORCH_CUDA_ARCH_LIST`, flash-attn compiled for sm_80 + sm_90 + sm_100 +
sm_120 (A100 + H100 + B100 + Blackwell). 72 CUDA kernel files × 4 archs = ~2.5 hours.

**Fix:** `TORCH_CUDA_ARCH_LIST="8.0"` — compiles for sm_80 only → ~35-40 min.

### Problem 4 — MAX_JOBS=4 on a 128-thread machine (slow)

Pod has 16 allocated vCPUs but `nproc` shows 128 hardware threads (dual AMD EPYC 7543).
Each `cicc` (CUDA compiler) process uses one thread and ~1 GB RAM. With 125 GB RAM and
128 threads, MAX_JOBS=4 leaves 96% of compute idle.

**Optimal:** `MAX_JOBS=16` (4× speedup, ~8-10 min, ~64 threads active — safe, won't freeze).
`MAX_JOBS=32` would use ~128 threads and ~32 GB RAM — possible but risks I/O contention.

### Problem 5 — Container disk full (30 GB)

The 6.2 GB venv (wrong torch 2.6+cu124 + unsloth + all deps) plus pip cache plus
flash-attn build artifacts filled the 30 GB container disk. torch 2.8.0 is ~2.3 GB —
there was no room to reinstall.

```
overlay  30G  29G  1.4G  96%  /             ← container disk, full
/workspace  700T  ...  /workspace            ← network volume, plenty of space
```

---

## What the Ideal Image Looks Like

| Requirement | Why |
|-------------|-----|
| torch ≥ 2.6, cu128 | Matches CUDA 12.8 toolkit in container, no ABI split |
| flash-attn pre-compiled | Avoids 15-40 min build every pod |
| Unsloth pre-installed | Avoids resolving transitive deps that break torch pin |
| Container disk ≥ 50 GB | venv + pip cache + build artifacts easily exceed 30 GB |
| OR venv on `/workspace` | Network volume has 100 GB; pip install to `/workspace/.venv` |
| Python 3.11 or 3.12 | Unsloth requirement |
| bitsandbytes compatible | 4-bit quantization for QLoRA |

---

## Research Results (2026-06-14)

### 1. Official Unsloth image — `docker.io/unsloth/unsloth:latest`

**CONFIRMED available on RunPod** (community template "Unsloth" exists).

From the README:
- Bundles: Unsloth, JupyterLab (port 8888), PyTorch+CUDA (GPU-ready), TRL, SFTConfig
- Supports: 4-bit, 8-bit, 16-bit, FP8 training paths
- Supports: Qwen, Llama, Gemma, DeepSeek, etc.
- Works with Blackwell (sm_100+)
- Working dir: `/workspace/work` (persistent volume)
- Example uses `SFTConfig` → confirms modern TRL included

**Unknown until we launch:** exact torch version, CUDA version, whether FA2 is bundled.
(Likely yes for FA2 since Unsloth uses it internally and they control the image.)

### 2. Community "LLM Ready" image — `konuu/llm_ready:latest`

From the README: "Pre-installed libraries: vLLM, SGLang, Unsloth, **flash-attn**,
flashinfer, Cloudflared". **Explicitly confirms FA2.**

Heavier image (includes vLLM/SGLang we don't need), less predictable versioning.

### 3. RunPod template "Unsloth-Finetune" — `docker.io/unsloth/unsloth`

Same official image, recommends:
- Container Disk: **50 GB**
- Volume Disk: 256 GB+
- "Works with Blackwell"

### 3. Confirmed Image Contents (from Dockerfile analysis)

| Component | Version |
|-----------|---------|
| Base | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` |
| torch | **2.10.0+cu128** |
| CUDA toolkit | **12.8.1** (nvcc present) |
| Python | **3.12** |
| unsloth | Latest (git install, pinned range) |
| bitsandbytes | >=0.49.2 |
| TRL | >=0.18.2, <=0.24.0 |
| transformers | >=4.51.3, <=5.5.0 |
| peft | >=0.18.0 |
| triton | >=3.6.0 |
| xformers | 0.0.34 (cu128 wheel) — **attention fallback** |
| vLLM | 0.16.0 |
| flash-attn | **NOT INCLUDED** |
| torchao | NOT included |
| nvcc + gcc + ninja | YES (can compile FA2) |
| JupyterLab | YES (port 8888) |
| llama.cpp | YES (prebuilt) |
| Image size | 13 GB compressed, ~25-30 GB uncompressed |
| GPU arch support | sm_75 through sm_120 (A100 sm_80 supported) |
| Container user | `unsloth` (may need --user root for volume perms) |

### 4. Decision: Use `docker.io/unsloth/unsloth:latest`

Why:
- Official → torch+unsloth+bitsandbytes+TRL compiled as a consistent unit
- Eliminates ALL 5 problems from our debugging history
- nvcc+gcc+ninja present → FA2 source build works cleanly against torch 2.10
- CUDA 12.8 toolkit is the container's native toolkit (no forward-compat hacks)
- Container disk: set to **100 GB** (not 30, not 50 — image + model + checkpoints)

### 5. Caveats

- **No FA2 pre-installed** — build from source (~10-30 min) or use xformers fallback
- **Python 3.12** — our scripts target 3.11 but should be compatible (test with smoke)
- **torch 2.10.0** — newer than what we used before, should be backward-compatible
- **TRL >=0.18.2** — uses `SFTConfig` path (our train_qlora.py already handles this)
- **User `unsloth`** — may need `--user root` in RunPod template for /workspace perms

### Setup script created: `setup_unsloth_pod.sh`

Validates pre-installed stack, installs only project deps (DVC, sklearn, xgboost),
handles FA2 source build (with S3 cache), runs smoke test. Total setup: ~5 min
without FA2 build, ~15-35 min with FA2 build (or 30 sec if S3-cached wheel exists).

---

## S3 Wheel Cache (already implemented)

While debugging, we built an S3 cache for compiled flash-attn wheels so future pods
don't recompile from scratch:

```
s3://trading-management-dvc/wheels/
  flash_attn-{FA_VER}-{TORCH_CUDA}-cp{PY}-sm{CC}-linux_x86_64.whl
  # e.g. flash_attn-2.8.3.post1-cu128-cp312-sm80-linux_x86_64.whl
```

`setup_runpod.sh` checks S3 first, builds + uploads on cache miss. Once we settle on
the right torch+CUDA combo, the first pod builds once (~10 min with MAX_JOBS=16) and
every subsequent pod gets a ~30-second install.

---

## Recommended Fix for setup_runpod.sh (independent of image choice)

```bash
# Replace nvidia-smi CUDA detection with nvcc-based detection:
if command -v nvcc &>/dev/null; then
    TOOLKIT_CUDA=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
else
    # Fallback: nvidia-smi (underestimates on RunPod forward-compat pods)
    TOOLKIT_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
fi
```

And raise MAX_JOBS to match available hardware:

```bash
# Instead of MAX_JOBS=4:
CPU_CORES=$(nproc)
MAX_JOBS=$(( CPU_CORES > 16 ? 16 : CPU_CORES ))  # cap at 16, safe on any machine
```

---

## Quick Decision Tree for Next Pod

```
Does the image have flash-attn pre-installed?
  YES → verify import works, skip build entirely
  NO  → does it have nvcc (devel image)?
          YES → build from source with FLASH_ATTENTION_FORCE_BUILD=TRUE
                TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=16
          NO  → wrong image, need devel variant

Is container disk ≥ 50 GB?
  YES → install venv normally at /trading_management/langgraph/.venv
  NO  → install venv at /workspace/.venv (network volume, persists across restarts)
```

---

## Commands to Validate a New Image (run immediately after pod start)

```bash
# 1. Verify CUDA toolkit matches torch
nvcc --version | grep release          # want: 12.8
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"

# 2. Check flash-attn (might already be there)
python3 -c "import flash_attn; print('FA2:', flash_attn.__version__)" 2>/dev/null \
    || echo "flash-attn not installed"

# 3. Disk space sanity check
df -h / /workspace

# 4. Thread count for MAX_JOBS decision
nproc
```
