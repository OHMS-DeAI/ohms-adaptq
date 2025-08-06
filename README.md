# OHMS-AdaptQ

**Adaptive Progressive Quantization (APQ) Engine**

Universal quantization pipeline that compresses diverse LLMs (Phi-3, Llama 3 8B, Mistral 7B, Gemma 2B/9B, Mixtral 8×7B) into 3-4-bit shards suitable for ICP canister storage and execution.

## Overview

APQ is the quantization engine that enables OHMS to run multiple model architectures on-chain. It produces deterministic artifacts that can be verified and loaded into ICP canisters.

### Key Features

- **Universal compatibility**: Works with Hugging Face transformer models
- **Progressive quantization**: Iterative 3-4-bit optimization with fidelity gates
- **Deterministic output**: Same input + config = identical artifacts
- **Sharded storage**: ≤2 MiB chunks for ICP message limits
- **Verification reports**: Per-layer error analysis and perplexity tracking

## Architecture

```
HF Model → APQ Pipeline → Artifacts Bundle
                      ↓
    ┌─ shards/ (≤2 MiB each)
    ├─ manifest.json
    ├─ model.meta  
    ├─ verification.json
    └─ LICENSE
```

## Artifacts

### Shards Directory
- Binary weight chunks, each ≤2 MiB
- SHA-256 hashed for integrity
- Indexed by manifest for loading

### manifest.json
```json
{
  "model_id": "phi-3-mini-4k",
  "version": "1.0.0",
  "chunks": [
    {
      "id": "chunk_000",
      "offset": 0,
      "size": 2097152,
      "sha256": "abc123..."
    }
  ],
  "digest": "global_manifest_hash"
}
```

### model.meta
```json
{
  "family": "phi",
  "arch": "microsoft/Phi-3-mini-4K-instruct",
  "tokenizer_id": "microsoft/Phi-3-mini-4K-instruct",
  "vocab_size": 32064,
  "ctx_window": 4096,
  "license": "mit"
}
```

### verification.json
```json
{
  "calibration_fingerprint": "calibration_hash",
  "layer_errors": [0.001, 0.002, ...],
  "perplexity_delta": 0.05,
  "OVERALL_STATUS": "PASS"
}
```

## APQ Method

1. **Calibration & Profiling**: Measure layer sensitivities with prompt set
2. **Progressive Rounding**: Iterative soft→hard quantization with error tracking
3. **Block Reconstruction**: Re-scale weights to minimize activation errors
4. **Dynamic Bit-Mix**: Selective 3-bit/4-bit allocation per sensitivity
5. **Fail-Safe Promotion**: Auto-upgrade low-res blocks if fidelity thresholds missed
6. **Sharding & Manifest**: Emit verified chunks with integrity hashes

## Acceptance Gates

- **Perplexity delta** within model-class budget vs FP16
- **Layer error histogram** shows no catastrophic outliers  
- **Functional spot-checks** on gold prompts within tolerance
- **Size budget** fits heap + cache plan after overhead
- **Determinism** identical artifacts under same seed/config

## Usage (CLI)

```bash
# Quantize a model
apq quantize microsoft/Phi-3-mini-4K-instruct \
  --output ./artifacts/phi-3-mini \
  --bits 3-4 \
  --calibration-size 512

# Verify artifacts
apq verify ./artifacts/phi-3-mini

# Generate reports
apq report ./artifacts/phi-3-mini --format json
```

## Integration

APQ artifacts are designed for:
- Upload to `ohms-model` canister (governance-gated)
- Loading by `ohms-agent` canisters (lazy, cached)
- Verification by auditors and governance voters

## Development

- **Language**: Rust (for deterministic quantization)
- **Dependencies**: HuggingFace transformers, candle-core
- **Testing**: Golden model comparisons, determinism checks
- **CI**: Artifact verification, hash consistency

## License

MIT - See LICENSE file

## Security

APQ outputs are deterministic and verifiable. All quantization is done with:
- Seeded randomness for calibration
- Reproducible bit allocation algorithms  
- Hash-verified integrity at every stage