# QLoRA Fine-Tuning — Qwen 2.5 7B

Diagrams explaining what fine-tuning produces, how the pipeline runs, and how the result plugs into production.

---

## 1. What fine-tuning actually is

```mermaid
flowchart LR
    BASE["Qwen 2.5 7B\n(base model)\n7B params\n'knows everything'"]
    DATA["Training data\n39,312 samples\nsystem / user / assistant\nchat format"]
    LORA["LoRA adapters\n~40M params\n0.6% of total\nonly these train"]
    FT["Fine-tuned\nQwen 2.5 7B\n= base + LoRA delta\n'knows crypto TA'"]

    BASE --> LORA
    DATA --> LORA
    LORA --> FT
```

The base model weights are **frozen** — only the small LoRA matrices are updated.
This is why it fits in 16 GB VRAM and trains in a few hours instead of weeks.

---

## 2. Full pipeline — one script, five steps

```mermaid
flowchart TD
    DS[("dataset.jsonl\n56,161 samples\n18 months · 12 symbols")]

    DS --> SPLIT["_temporal_split\n70 / 15 / 15\nno shuffle\nsame split as LSTM/XGBoost"]

    SPLIT --> TRAIN_D["train.jsonl\n39,312 samples"]
    SPLIT --> VAL_D["val.jsonl\n8,424 samples"]
    SPLIT --> TEST_D["test.jsonl\n8,425 samples\n(held out)"]

    TRAIN_D --> STEP1

    subgraph STEP1["① prepare_data()"]
        direction LR
        FMT["Convert each sample to\nsystem · user · assistant\nchat messages"]
    end

    STEP1 --> STEP2

    subgraph STEP2["② load_model()  ~5 min"]
        direction LR
        Q["Qwen2.5-7B-Instruct-bnb-4bit\nfrom Unsloth HuggingFace Hub\n~4 GB download · 4-bit quant"]
        L["Apply LoRA adapters\nrank=16  alpha=32\ntarget: q/k/v/o + gate/up/down proj"]
        Q --> L
    end

    STEP2 --> STEP3

    subgraph STEP3["③ train()  ~90-150 min on A10G"]
        direction LR
        T["SFTTrainer\n3 epochs\nbatch=4 × grad_accum=4 → effective 16\ncosine LR  warmup=50 steps"]
        E["eval every 100 steps\non val.jsonl\nsave best checkpoint\nby eval_loss"]
        T --> E
    end

    STEP3 --> STEP4

    subgraph STEP4["④ save_model()  ~15 min"]
        direction LR
        AD["adapter_model.safetensors\n~80 MB\nfor research / retraining"]
        GG["unsloth.Q4_K_M.gguf\n~4 GB\nfor Ollama production"]
        AD --- GG
    end

    STEP4 --> STEP5

    subgraph STEP5["⑤ evaluate()  ~2-4h on GPU"]
        direction LR
        P["Rebuild prompts from\nraw indicators\n(not from training JSONL)"]
        G["Generate prediction\ngreedy decoding\ndo_sample=False"]
        PS["Parse full JSON\nbias + entry + tp + sl\nfallback on error — no skip"]
        SIM["simulate_trade\nagainst future candles\nWIN / LOSS / TIMEOUT"]
        P --> G --> PS --> SIM
    end

    TEST_D --> STEP5

    STEP5 --> OUT

    subgraph OUT["optimization/results/qlora_optimization.json"]
        direction LR
        M["metrics\naccuracy · win_rate\nprofit_factor · Sharpe\nmax_drawdown"]
        PER["per-sample arrays\npredictions · actuals\ntrade_results · sample_keys"]
        M --- PER
    end
```

---

## 3. Output files

```mermaid
flowchart LR
    subgraph DISK["backtest/data/models/qlora_qwen25_7b/"]
        AC["adapter_config.json\n~1 KB"]
        AM["adapter_model.safetensors\n~80 MB\nthe trained LoRA delta"]
        TK["tokenizer files\n~7 MB"]
        subgraph GGUF_DIR["gguf/"]
            GF["unsloth.Q4_K_M.gguf\n~4 GB\nbase + LoRA fused\n4-bit quantized"]
        end
    end

    subgraph RESULTS["optimization/results/"]
        RJ["qlora_optimization.json\nfull metrics + predictions\nfor thesis comparison"]
    end
```

**Only the GGUF goes to production.** The adapters are kept for future retraining rounds.

---

## 4. How the result plugs into production

```mermaid
flowchart TD
    subgraph BEFORE["Before fine-tuning"]
        B_ENV["LLM_MODEL=qwen2.5:14b"]
        B_MOD["Qwen 2.5 14B\n~10 GB VRAM\nzero-shot general"]
        B_ENV --> B_MOD
    end

    subgraph DEPLOY["Deploy fine-tuned model (one-time)"]
        MF["Write Modelfile:\nFROM unsloth.Q4_K_M.gguf\nPARAMETER temperature 0.1"]
        OC["ollama create trading-qwen-ft -f Modelfile"]
        MF --> OC
    end

    subgraph AFTER["After fine-tuning"]
        A_ENV["LLM_MODEL=trading-qwen-ft"]
        A_MOD["Qwen 2.5 7B fine-tuned\n~5 GB VRAM\ndomain-specific crypto TA"]
        A_ENV --> A_MOD
    end

    BEFORE --> DEPLOY --> AFTER

    AFTER --> LLM["llm_factory.py\nOllamaProvider\nno code changes needed\nreads LLM_MODEL from env"]
    LLM --> GEN["generator_node()\nLangGraph agent\nsame interface, better results"]
```

---

## 5. Why this matters for the thesis

```mermaid
flowchart LR
    subgraph ENFOQUE1["Approach 1\nZero-shot LLM\nLlama 3.3 70B · Groq API"]
        E1["+ Strong reasoning\n+ No training cost\n− API tokens\n− No domain adaptation"]
    end

    subgraph ENFOQUE2["Approach 2\nReAct Agent\nLangGraph 3-node DAG"]
        E2["+ Structured reasoning\n+ Deterministic evaluator\n− Still zero-shot\n− Latency (3 nodes)"]
    end

    subgraph ENFOQUE3["Approach 3\nML Models\nLSTM · XGBoost · RF"]
        E3["+ Fast inference\n+ No GPU for inference\n− No reasoning output\n− Feature engineering"]
    end

    subgraph ENFOQUE4["Approach 4  ← this script\nFine-tuned LLM\nQwen 2.5 7B · QLoRA"]
        E4["+ Domain-specific\n+ Local · no API tokens\n+ Smaller · faster than 14B\n+ Explainable reasoning\n− Training cost (one-time)"]
    end

    ENFOQUE1 --> CMP
    ENFOQUE2 --> CMP
    ENFOQUE3 --> CMP
    ENFOQUE4 --> CMP

    CMP{"compare-stats\nMcNemar test\npaired t-test\nsame 8,425 test samples"}

    CMP --> Q["Research question:\nDoes fine-tuning significantly improve\nLLM accuracy for crypto TA\nwhile maintaining explainable reasoning?"]
```

---

## 6. Zero-shot vs fine-tuned — what changes

```mermaid
flowchart TD
    IND["Multi-TF indicators\nBTC · RSI=42 · ADX=31\nEMA aligned · MACD bullish"]

    subgraph ZS["Zero-shot Qwen 2.5 14B"]
        ZP["Generic system prompt\n'You are a trading analyst...'"]
        ZR["Has to infer domain rules\nfrom the prompt alone\nno prior exposure to this task"]
        ZO["Output JSON\nbias · entry · tp · sl\nreasoning (generic)"]
        ZP --> ZR --> ZO
    end

    subgraph FT["Fine-tuned Qwen 2.5 7B"]
        FP["Same system prompt"]
        FR["Has seen 39,312 examples\nof correct setups\nalready knows RSI thresholds,\nATR-based SL sizing, etc."]
        FO["Output JSON\nbias · entry · tp · sl\nreasoning (domain-specific)"]
        FP --> FR --> FO
    end

    IND --> ZS
    IND --> FT

    ZO --> ACC["Accuracy compared\non the SAME 8,425 test samples\n→ McNemar p-value"]
    FO --> ACC
```
