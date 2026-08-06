# Plato Server Facts and Operating Envelope

**Status:** verified from server information supplied by Prof. Markus Schaal  
**Recorded:** 2026-08-06

## Source-derived server facts

The account has:

- an initial storage allocation of 20 GB;
- no hard implementation of that initial limit at present;
- temporary permission to use up to 80 GB;
- the option to request more storage for a limited period when required by a research project;
- access, shared equally among users, to 1.5 TB RAM;
- access, shared equally among users, to four NVIDIA Tesla V100 GPUs with 32 GB VRAM each;
- a current Lambda Stack containing TensorFlow, PyTorch, CUDA and cuDNN;
- the expectation that users inspect current load with `htop` and `nvidia-smi` and use resources carefully and considerately;
- a recommendation to use a personal Python virtual environment;
- a documented Python 3.8 virtualenv route for exposing preinstalled Python packages;
- Remote SSH access through the university network or HWR VPN.

No scheduler, reservation system or automatic interest-management mechanism is described in this server information.

## Consequences for this project

### Immediate smoke

The Qwen3.5-0.8B one-GPU smoke remains compatible with the stated server envelope, subject to the measured storage preflight, successful CUDA build and actual model load.

The project uses isolated environments and pinned dependencies rather than relying on mutable system Python packages. The documented Python 3.8 virtualenv route remains available as a fallback or diagnostic path, but the current harness intentionally uses its pinned Python 3.12 worker environment and separate Python 3.11 ContextBench evaluator environment.

The statement that the Lambda Stack contains CUDA makes an installed CUDA toolkit plausible, but it does not by itself verify that `nvcc` is available at the path expected by the pinned llama.cpp build. `nvcc` must still be checked directly.

### Storage

The 80 GB temporary allowance is an administrative operating ceiling, not a guaranteed filesystem quota. Both actual filesystem space and home-directory usage must be checked.

The provisioning pipeline may temporarily hold three large representations:

1. the exact Hugging Face source snapshot;
2. an F16 GGUF conversion intermediate;
3. the final Q8_0 GGUF.

This peak can exceed the size of the final artifact by a large factor. The provisioning command therefore performs a conservative storage estimate, writes the final artifact atomically, and removes the source snapshot and F16 intermediate after successful conversion unless explicitly instructed to retain them.

For the largest checkpoints, the temporary 80 GB allowance may be insufficient for local source-to-Q8 conversion. The approved resolution paths are:

- request additional temporary storage from the server administrator;
- perform conversion on another machine with sufficient storage and register/copy only the final hashed Q8_0 artifact;
- provision and execute model waves sequentially, retaining only the artifacts required by the current wave.

The model family or scientific scope must not be silently reduced because of storage constraints.

### Shared compute behavior

The four GPUs and 1.5 TB RAM are shared equally. Prof. Eck separately permits computation until he contacts the user. Plato therefore remains in `manual_operator` mode:

- inspect `htop` and `nvidia-smi` before starting;
- start only the intended worker count;
- monitor the run;
- use central pause and graceful stop when resources must be released;
- do not claim scheduler preemption that the server does not provide.

## Storage checks

Before bootstrap or provisioning on Plato:

```bash
python3 scripts/storage_preflight.py \
  --soft-limit-gib 80 \
  --minimum-filesystem-free-gib 5
```

The provisioning command performs an additional conversion-specific free-space estimate. For a larger approved storage allocation, pass the new administrative ceiling to the preflight rather than continuing to use 80 GB.
