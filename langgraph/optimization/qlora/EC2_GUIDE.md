# QLoRA Fine-Tuning on EC2 (Deep Learning AMI)

Step-by-step guide to fine-tune Qwen 2.5 7B on a raw **EC2 g6e.xlarge** (NVIDIA
L40S 48GB). EC2 + the Deep Learning AMI is the recommended path: it's **cheaper**
than SageMaker *and* FlashAttention-2 actually works (matched CUDA), which roughly
halves training time.

> Same hardware as the old SageMaker guide (L40S 48GB), but on raw EC2 so FA2 builds
> and you keep root/SSH/`nohup`. The single setup script `setup_ec2.sh` also works on
> a SageMaker Studio terminal if you ever need it (FA2 just won't build there).

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

## Step 0: Pick the right AMI + instance

| Field | Value |
|-------|-------|
| **AMI** | `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)` — search "Deep Learning OSS" in the AMI catalog, pick the newest **PyTorch** one. It ships a **matched** NVIDIA driver + CUDA toolkit, so FA2 compiles. |
| **Instance** | `g6e.xlarge` — L40S 48GB, Ada Lovelace `sm_89`, ~$1.86/hr on-demand |
| **Storage** | 100 GB gp3 |
| **Key pair** | your existing SSH key |
| **Security group** | inbound SSH (22) from **your IP only** |
| **IAM role** | attach a role with `s3:GetObject` on `s3://trading-management-dvc/` (lets DVC pull without `aws configure`) |

> ⚠️ Choose the **OSS Nvidia Driver** AMI, not the "Base" AMI (Base has no driver).
> For L40S (Ada) any DLAMI with CUDA ≥ 12.1 works; recent ones ship 12.4+.

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
builds FlashAttention-2, installs DVC, and **verifies versions** (driver, CUDA,
PyTorch, FA2, compute capability). Expected output on L40S:

```
GPU: NVIDIA L40S  cc=(8, 9)
CUDA (torch): 12.x   PyTorch: 2.x
FlashAttention-2: v2.x  (expect FA2=True at train time)
```

> **Confirm `FA2 = True`** in the Unsloth startup banner when the model loads — that's
> the whole speedup. If it's `False`, see Troubleshooting.

---

## Step 3: Pull data (REQUIRED)

```bash
source .venv/bin/activate
dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
```

Without the candles, trade simulation has no data → `win_rate`/`profit_factor`/
`Sharpe` come out 0 (direction accuracy still works).

---

## Step 4: Smoke test (~2 min, ~$0.06)

```bash
python optimization/qlora/train_qlora.py --max-steps 3 --max-eval 10
```

Loads the model, runs 3 steps, evaluates 10 samples, exits. If step 0 passes, the
full run will too — the failure mode is always step 0.

---

## Step 5: Run the real training (nohup)

`nohup` keeps the job alive if SSH drops (tmux works on EC2 too, but nohup is the
portable default we standardize on):

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
3. **train** — SFTTrainer, 3 epochs, `batch_size=2 × grad_accum=8`, cosine LR,
   eval every 250 steps, **early stopping (patience 3)**, keeps best by eval_loss
4. **save_loss_curve** — `results/qlora_cloud_loss_curve.{json,png}`
5. **save_model** — LoRA adapters + merged GGUF (Q4_K_M) for Ollama
6. **evaluate** — greedy decoding on the FULL test split, trade simulation,
   train/val/test accuracy + gap, heuristic baseline, `sample_keys` for paired tests
7. **save result** → `results/qlora_cloud.json` (+ canonical `qlora_optimization.json`)

### Batch size note
`batch_size=2` is the safe L40S ceiling. With FA2 confirmed you can try
`--batch-size 4` (FA2 lowers the backward-pass VRAM peak) after a `--max-steps 3`
smoke test at the higher batch. Without FA2, batch 4 OOMs.

### If interrupted
```bash
nohup bash optimization/qlora/run_cloud.sh --resume > optimization/qlora/logs/run_cloud.log 2>&1 &
```
`--resume` continues from the last `checkpoint-N` (saved every 250 steps). No work lost.

### Expected timing (g6e.xlarge, FA2 on)
~4–5 s/step · ~1.5–2.5 h/epoch · **~5–7 h** for 3 epochs + eval.

---

## Step 6: Graph the overfitting dashboard

```bash
python optimization/qlora/plot_overfitting.py --result optimization/qlora/results/qlora_cloud.json
# → results/qlora_cloud_overfitting.png (loss curve, train/val/test gap, vs baseline, per-symbol)
```

---

## Step 7: Back up the model + STOP THE INSTANCE

```bash
# Back up adapters + GGUF to S3 (model weights are too large for git)
aws s3 cp --recursive backtest/data/models/qlora_cloud/ s3://trading-management-dvc/models/qlora_cloud/

# Commit just the result JSON
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
python optimization/qlora/train_qlora.py --lr 0.00002 --rank 16 --alpha 32 \
  --epochs 3 --batch-size 2 --grad-accum 8 --tag qlora_cloud
```
