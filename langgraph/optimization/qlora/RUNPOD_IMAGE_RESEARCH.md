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

## Research Questions (for the next session / agent)

### 1. Official Unsloth image
- Does `unsloth/unsloth` exist on Docker Hub with flash-attn pre-compiled?
- Check: `docker pull unsloth/unsloth` — what torch/CUDA/FA2 versions?
- RunPod template search: does a community template "Unsloth" or "QLoRA" exist?

### 2. RunPod's own torch images
- What is the exact Docker Hub tag behind `runpod-torch-v280`?
  (Likely `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04`)
- Does any RunPod image include flash-attn? Check `runpod/pytorch` tags on Docker Hub.
- Is there a `-devel` vs `-runtime` split? (devel has nvcc, runtime doesn't — we need devel for FA2 compilation)

### 3. Using `/workspace` as venv location
- Can we do `python -m venv /workspace/.venv` and install everything there?
- Does RunPod's network volume have enough IOPS for pip installs? (It's a distributed FS)
- Would the venv persist across pod restarts if on `/workspace`?

### 4. NVIDIA NGC containers
- `nvcr.io/nvidia/pytorch:24.xx-py3` ships torch + CUDA + flash-attn pre-compiled
- Check latest tag compatible with A100 (sm_80) + bitsandbytes + Unsloth
- Disk footprint of NGC containers (they're large, ~20-30 GB image)

### 5. Container disk size
- Can RunPod pods be launched with larger container disk (50-100 GB)?
- Or is the right pattern to always install to `/workspace` and keep container disk minimal?

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
