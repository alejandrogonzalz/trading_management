# QLoRA Fine-Tuning on EC2 (Deep Learning AMI)

Step-by-step guide to fine-tune Qwen 2.5 7B on a raw **EC2 g6e.xlarge** (NVIDIA
L40S 48GB). EC2 + the Deep Learning AMI is the recommended path: it's **cheaper**
than SageMaker *and* FlashAttention-2 actually works (matched CUDA), which roughly
halves training time.

> Same hardware as the old SageMaker guide (L40S 48GB), but on raw EC2 so FA2 builds
> and you keep root/SSH/`nohup`. The single setup script `setup_ec2.sh` also works on
> a SageMaker Studio terminal if you ever need it (FA2 just won't build there).
>
> **Prefer RunPod?** It's often cheaper and faster (FA2 works, A100/H100 by the
> minute). Jump to [RunPod (alternative to EC2)](#runpod-alternative-to-ec2--usually-cheaper-and-faster)
> for an instance-by-instance time/cost table — the scripts and steps are the same.

---

## Cost analysis (this is why EC2)

Cost of **one** model, 3 epochs + evaluation, on the same L40S 48GB GPU:

| | EC2 g6e.xlarge (DLAMI) | SageMaker ml.g6e.xlarge |
|---|---|---|
| On-demand $/hr | **~$1.86** | ~$2.00–2.35 |
| FlashAttention-2 | ✅ builds (matched CUDA) | ❌ container CUDA mismatch |
| Throughput | ~4–5 s/step (FA2) | ~8.7 s/step (no FA2) |
| Time, 3 epochs | **~5–7 h** | ~14 h |
| **Compute cost (run)** | **~$9–13** | ~$28–33 |
| tmux / SSH / root | ✅ | ❌ (nohup only, restricted) |
| Auto-shutdown | ⚠️ you must stop it | idle lifecycle config |

> Net: EC2 is roughly **half the time and ~⅓ the cost** for the same GPU, because
> FA2 works. The one tradeoff — you must remember to `stop` the instance (Step 7).

Spot pricing can cut the on-demand rate ~60–70% more, but spot can be reclaimed
mid-run; for a single ~6h job, on-demand + remembering to stop is simpler.

Storage: a stopped instance keeps its EBS volume (~$8/mo for 100GB gp3); you only
pay GPU while it's **running**. `terminate` deletes everything.

---

## RunPod (alternative to EC2 — usually cheaper *and* faster)

RunPod rents the same class of GPUs by the minute, often **below EC2/SageMaker
on-demand**, and because it's a Linux container on cu128/torch 2.8,
**FlashAttention-2 works** just like on the DLAMI. You launch a *pod* from a Docker
image instead of an AMI; the rest of the flow (`setup_ec2.sh` → `dvc pull` → smoke
test → `run_cloud.sh`) is identical.

**Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (torch 2.8 + CUDA 12.8).

### Which instance? (estimated: ONE model, 2 epochs + 3k strided eval)

Timings are **estimates anchored on the measured L40S numbers** (8.7 s/step without
FA2; ~17 s/generate serial) scaled by compute/bandwidth, FA2, and the **new batched
eval**. With batching, evaluation is minutes — so **training is now the only long
pole**. All rows keep the effective batch at 16 (`BATCH×GA`), so the 2e-5 LR stays
valid.

| GPU (RunPod) | VRAM | $/hr | `BATCH/GA` | `EVAL_BATCH` | Train (2 ep) | Eval (3k strided) | **Total** | **Run cost** |
|---|---|---|---|---|---|---|---|---|
| **A100 80GB** ⭐ | 80GB | $1.39 | `8/2` | `32` | ~3–3.5 h | ~5–8 min | **~3.5–4 h** | **~$5–6** |
| H100 SXM (fastest) | 80GB | $3.29 | `8/2` | `32` | ~1.5–2.5 h | ~3–5 min | **~2–2.5 h** | ~$7–8 |
| L40S (= EC2 guide HW) | 48GB | $0.86 | `4/4` | `16` | ~6–7 h | ~8–12 min | ~6–7 h | ~$5–6 |
| RTX 4090 | 24GB | $0.69 | `2/8` | `8` | ~5–6 h | ~12–18 min | ~6 h | ~$4 |
| RTX A6000 / A40 | 48GB | $0.49 / $0.44 | `4/4` | `16` | ~12–14 h | ~10–15 min | ~12–14 h | ~$6–7 |

> **Pick A100 80GB** — best cost-and-speed balance (~$5, under 4 h). Pick **H100 SXM**
> if you want it done in ~2 h and don't mind ~$7. The cheap Ampere 48GB cards
> (A6000/A40) have the lowest $/hr but slow GDDR6 → long wall-clock; only worth it if
> you're not time-pressed. **Avoid Blackwell (B200 / RTX 5090)** — bitsandbytes 4-bit
> kernels are still flaky there, not worth debugging on a deadline.
>
> For **3 epochs** instead of 2, multiply the training column by ~1.5 (A100 ~4.5–5 h,
> ~$7). The eval column is unchanged — it doesn't depend on epoch count.

### Launch + run (deltas from the EC2 steps below)
1. **Pod**: choose a GPU above, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`,
   and attach a **persistent network volume** mounted at `/workspace` so the adapters
   + GGUF survive a pod stop. Enable SSH.
2. **Setup**: same `bash optimization/qlora/setup_ec2.sh` (builds an isolated venv; lets
   Unsloth resolve torch/TRL — a torch re-pin inside the venv is harmless). Confirm the
   banner shows `FA2 = True` (Linux/cu128 builds it, unlike Windows).
3. **S3 creds**: no IAM roles on RunPod → run `aws configure` (or export
   `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`) before `dvc pull` and the S3 backup.
4. **Run** (A100 example — overrides the L40S-safe defaults):
   ```bash
   tmux new -s qlora            # persistent pods have tmux; nohup also works
   source .venv/bin/activate
   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles   # REQUIRED
   # smoke test the chosen batch sizes first (confirms no OOM + batched eval engages):
   BATCH=8 EVAL_BATCH=32 python optimization/qlora/train_qlora.py \
     --max-steps 3 --max-eval 20 --eval-batch-size 32 --diagnostic-samples 0
   # then the real run:
   BATCH=8 GRAD_ACCUM=2 MAX_EVAL=3000 EVAL_BATCH=32 \
     bash optimization/qlora/run_cloud.sh 2>&1 | tee optimization/qlora/logs/run_cloud.log
   ```
   In the smoke-test log, confirm `batch_size=32, … batches` with **no** "batched
   generate failed" warnings — that means batching actually engaged (not silently
   falling back to the slow serial path).
5. **Stop the pod** when done (RunPod bills per-minute while running; a stopped pod
   keeps only the volume).

The EC2-specific steps below (AMI choice, IAM, `stop-instances`) don't apply to
RunPod, but **Steps 2–8 are otherwise identical** — same scripts, same flags.

---

## Step 0: Pick the right AMI + instance

| Field | Value |
|-------|-------|
| **AMI** | `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 24.04)` — search "Deep Learning OSS" in the AMI catalog, pick the newest **PyTorch** one. It ships a **matched** NVIDIA driver + CUDA toolkit, so FA2 compiles. |
| **Instance** | `g6e.xlarge` — L40S 48GB, Ada Lovelace `sm_89`, ~$1.86/hr on-demand |
| **Storage** | 100 GB gp3 |
| **Key pair** | your existing SSH key |
| **Security group** | inbound SSH (22) from **your IP only** |
| **IAM role** | attach a role with `s3:GetObject` on `s3://trading-management-dvc/` (lets DVC pull without `aws configure`) |

> ⚠️ Choose the **OSS Nvidia Driver** AMI, not the "Base" AMI (Base has no driver).
> For L40S (Ada) any DLAMI with CUDA ≥ 12.1 works; recent ones ship 12.4+.
>
> **Note:** on the DLAMI, `nvidia-smi` lives under `/opt/pytorch/bin/` — not in the
> default PATH. The setup script adds it to `~/.bashrc` automatically, but on the
> very first SSH login (before running setup) you may need `source /opt/pytorch/bin/activate`
> or `export PATH="/opt/pytorch/bin:$PATH"` to see the GPU.

### Optional: launch from the AWS CLI

Faster than clicking through the console. Replace every `<...>` placeholder and run
in your target region (add `--region us-east-1`). Find the AMI id with:

```bash
aws ec2 describe-images --owners amazon \
  --filters 'Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu*' \
  --query 'reverse(sort_by(Images,&CreationDate))[:3].[ImageId,Name]' --output table
```

```bash
# 1. Security group (SSH in). Prefer YOUR_IP/32 over 0.0.0.0/0 (open to the world).
SG_ID=$(aws ec2 create-security-group \
  --group-name qlora-finetuning \
  --description "QLoRA fine-tuning SSH access" \
  --vpc-id <your-vpc-id> \
  --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress \
  --group-id "$SG_ID" \
  --ip-permissions '{"IpProtocol":"tcp","FromPort":22,"ToPort":22,"IpRanges":[{"CidrIp":"<YOUR_IP>/32"}]}'

# 2. Launch: g6e.xlarge, DLAMI, 100GB gp3 root, public IP. (SnapshotId is inherited
#    from the AMI — don't hardcode it.) --key-name is your existing EC2 key pair.
aws ec2 run-instances \
  --image-id <dlami-ami-id> \
  --instance-type g6e.xlarge \
  --key-name <your-key-pair> \
  --block-device-mappings '{"DeviceName":"/dev/sda1","Ebs":{"DeleteOnTermination":true,"VolumeSize":100,"VolumeType":"gp3"}}' \
  --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["'"$SG_ID"'"]}' \
  --count 1
```

> `<your-vpc-id>` → your default VPC (`aws ec2 describe-vpcs --query 'Vpcs[?IsDefault].VpcId'`).
> `<YOUR_IP>` → `curl -s ifconfig.me`. `<your-key-pair>` → the name of a key pair you
> already own (needed to SSH in). Attach the DVC IAM role with
> `--iam-instance-profile Name=<your-profile>` if you use one.

---

## Step 1: Connect + clone

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<public-ip>

git clone git@github.com:luisaga215/trading_management.git
cd trading_management/langgraph
```

---

## Step 2: Setup (one command, ~10–15 min)

```bash
bash optimization/qlora/setup_ec2.sh
```

It creates a clean venv, installs Unsloth (which resolves its own torch/TRL),
installs DVC + tmux, and **verifies versions** (driver, CUDA, PyTorch, Unsloth).
Expected output on L40S:

```
GPU: NVIDIA L40S  cc=(8, 9)
CUDA (torch): 13.x   PyTorch: 2.x
FlashAttention-2: NOT INSTALLED — training will be ~2-3x slower
Unsloth: OK
```

> FA2 is skipped by default (compilation freezes 4-vCPU instances). Training still
> works via Unsloth's Triton kernels (~8.7s/step instead of ~4-5s). See comments in
> `setup_ec2.sh` for manual FA2 install instructions.

---

## Step 3: Pull data (REQUIRED)

```bash
source .venv/bin/activate
dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
```

Without the candles, trade simulation has no data → `win_rate`/`profit_factor`/
`Sharpe` come out 0 (direction accuracy still works).

---

## Step 4: Smoke test (~5 min, ~$0.15)

```bash
python optimization/qlora/train_qlora.py --max-steps 3 --max-eval 10 --diagnostic-samples 0 \
  2>&1 | tee optimization/qlora/logs/smoke_test.log
```

Loads the model, trains 3 steps, evaluates 10 samples, generates GGUF, exits.
Uses `--diagnostic-samples 0` to skip the 500+500 train/val accuracy probes (those
add ~4h to a smoke test). If step 0 passes and you see `RESULT:` at the end with
`Win Rate > 0`, the full run will work.

---

## Step 5: Run the real training

### Option A: tmux (recommended)

tmux keeps the job alive if SSH drops **and** lets you reattach to see live output:

```bash
tmux new -s qlora
source .venv/bin/activate
bash optimization/qlora/run_cloud.sh 2>&1 | tee optimization/qlora/logs/run_cloud.log
```

- **Detach** (leave running): `Ctrl+B`, then `D`
- **Reattach** (see live output): `tmux attach -t qlora`
- **Monitor from another SSH**: `tail -f optimization/qlora/logs/run_cloud.log`

### Option B: nohup (simpler, no reattach to live output)

```bash
nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &
echo "PID: $!"

# Monitor:
tail -f optimization/qlora/logs/run_cloud.log
nvidia-smi
```

### What `run_cloud.sh` does
1. **prepare_data** — strict temporal split (global timestamp sort + embargo) →
   `training_data/{train,val,test}.jsonl` (same split as LSTM/XGBoost)
2. **load_model** — Qwen 2.5 7B in 4-bit (~4-5 GB VRAM) + LoRA adapters
3. **train** — SFTTrainer, 2 epochs, `batch_size=2 × grad_accum=8`, cosine LR,
   eval every 250 steps, **early stopping (patience 3)**, keeps best by eval_loss
4. **save_loss_curve** — `results/qlora_cloud_loss_curve.{json,png}`
5. **save_model** — LoRA adapters + merged GGUF (Q4_K_M) for Ollama
6. **evaluate** — greedy decoding on `MAX_EVAL` test samples **strided across the
   full holdout** (default 3000 ≈ 270/symbol, spans the whole period — not a prefix),
   **batched via `EVAL_BATCH`** so it takes minutes not hours; trade simulation,
   train/val/test accuracy + gap, heuristic baseline, `sample_keys`. Set `MAX_EVAL=`
   (empty) to evaluate all 8,425.
7. **save result** → `results/qlora_cloud.json` (+ canonical `qlora_optimization.json`)

### Batch / eval-batch note
Training: `BATCH=2` is the safe L40S ceiling without FA2 (batch 4 OOMs); with FA2 use
`BATCH=8 GRAD_ACCUM=2` (keeps effective batch 16). Eval: `EVAL_BATCH` is independent
(no gradients) — 16 is safe on 48GB, 32 on 80GB; it's the knob that turns a serial
multi-hour eval into minutes. Both are env vars on `run_cloud.sh` (see the RunPod
table for per-GPU values).

### If interrupted
```bash
tmux attach -t qlora   # if using tmux, just reattach — it's still running

# If the process died (SSH drop without tmux, or Ctrl+C):
bash optimization/qlora/run_cloud.sh --resume 2>&1 | tee optimization/qlora/logs/run_cloud.log
```
`--resume` continues from the last `checkpoint-N` (saved every 250 steps). No work lost.

### Expected timing
- **L40S without FA2** (raw g6e default): ~8.7 s/step · ~12 h training (2 epochs) ·
  eval ~10–25 min (3000 strided, `EVAL_BATCH=16`) · **~12 h total** (eval is no longer
  the bottleneck — it used to be ~6 h serial).
- **L40S with FA2** (built): ~4–5 s/step · ~6 h total.
- **A100 / H100 (RunPod, FA2 + `BATCH=8`)**: **~2–4 h total** — see the RunPod
  instance table above for per-GPU numbers and cost.

---

## Step 6: Graph the overfitting dashboard

```bash
python optimization/qlora/plot_overfitting.py --result optimization/qlora/results/qlora_cloud.json
# → results/qlora_cloud_overfitting.png (loss curve, train/val/test gap, vs baseline, per-symbol)
```

---

## Step 7: Back up the model + STOP THE INSTANCE

```bash
# Track model weights with DVC (too large for git — adapters ~80MB + GGUF ~4GB)
dvc add backtest/data/models/qlora_cloud
dvc push   # uploads to s3://trading-management-dvc/

# Commit the DVC pointer + result JSON
git add backtest/data/models/qlora_cloud.dvc backtest/data/models/.gitignore
git add optimization/qlora/results/qlora_cloud.json
git commit -m "feat(qlora): fine-tuning results on fixed split (EC2 g6e.xlarge)"
git push origin <branch>
```

**Then STOP the instance** (you pay GPU only while running ≈ $1.86/hr ≈ $45/day):

```bash
# From your laptop:
aws ec2 stop-instances --instance-ids i-xxxxxxxxxxxx --region us-east-1
# Restart later (public IP changes — use an Elastic IP if you want it fixed):
aws ec2 start-instances --instance-ids i-xxxxxxxxxxxx --region us-east-1
```

`stop` keeps the disk (no GPU charge). `terminate` deletes everything.

---

## Step 8: Compare results

```bash
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_optimization.json \
  --b backtest/data/results/ml-lstm.json
```

---

## Optional: deploy to Ollama

```bash
# Copy the GGUF off the box (scp or via the S3 backup), then:
ollama create trading-qwen-ft -f Modelfile
```

`Modelfile`:
```
FROM ./qlora_cloud/gguf/unsloth.Q4_K_M.gguf
PARAMETER temperature 0.1
SYSTEM "You are a Senior Technical Analyst..."
```
Then set `LLM_MODEL=trading-qwen-ft` — the agent picks it up via `llm_factory.py`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FA2 = False` in banner | `pip install flash-attn --no-build-isolation` inside the venv; ensure you're on the **OSS Nvidia Driver** DLAMI (matched CUDA), not the Base AMI |
| CUDA OOM | Drop `--batch-size` to 2 (or 1), raise `--grad-accum` proportionally to keep effective batch |
| Slow training / low GPU util | `nvidia-smi`: if util < 80%, try `--batch-size 4` (with FA2) |
| Interrupted mid-training | Re-run with `--resume` |
| DVC pull fails | Attach an IAM role with `s3:GetObject` on the bucket, or `aws configure` |
| Instance won't launch | Request a service quota increase for `g6e.xlarge` (vCPU) in your region |
| SSH drops kill the job | You launched without `nohup` — always use the `nohup ... &` form (Step 5) |
| Forgot to stop it | `aws ec2 stop-instances ...` from your laptop; consider a CloudWatch idle alarm |

---

## Versions to verify (performance)

**Cloud — L40S (Ada `sm_89`)** is mature hardware; almost any recent DLAMI works.
Just confirm `FA2 = True` and `get_device_capability() == (8, 9)`.

Verified working configuration (Jun 2026):

| Component | Version |
|-----------|---------|
| AMI | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Ubuntu 24.04) |
| Driver | 595.71.05 |
| CUDA | 13.2 |
| GPU | NVIDIA L40S, 46068 MiB |

**Local — RTX 5070 Ti (Blackwell `sm_120`)** is *new* hardware; versions are
make-or-break (see `run_local.sh`):

| Component | Minimum for Blackwell | Why |
|-----------|------------------------|-----|
| NVIDIA driver | ≥ R570 (Windows ~572.xx+) | `sm_120` absent in older drivers |
| PyTorch | ≥ 2.7, **cu128** build | first stable with `sm_120` kernels |
| CUDA (in torch) | 12.8 (bundled in cu128 wheel) | Blackwell supported from CUDA 12.8 |
| bitsandbytes | ≥ 0.45 | 4-bit on `sm_120` added there |
| Triton | recent `triton-windows` | Unsloth uses Triton (no FA2 on Windows) |

Verify: `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"`
→ want `2.7+ 12.8 (12, 0)`.

---

## Single model, not a sweep

The thesis trains **one** defensible model on the fixed strict-temporal split — the
old multi-config sweep was removed (results archived under `results/archive/`, must
not be cited). `run_cloud.sh` is the wrapper; to invoke directly:

```bash
# L40S-safe defaults; on an 80GB A100/H100 use --batch-size 8 --grad-accum 2
python optimization/qlora/train_qlora.py --lr 0.00002 --rank 16 --alpha 32 \
  --epochs 3 --batch-size 2 --grad-accum 8 \
  --max-eval 3000 --eval-batch-size 16 --tag qlora_cloud
```
> `--max-eval` strides across the full holdout (representative); omit it to eval all
> 8,425. `--eval-batch-size` batches generation (minutes vs hours). `run_cloud.sh`
> wraps this with the `BATCH/GRAD_ACCUM/MAX_EVAL/EVAL_BATCH` env vars.
