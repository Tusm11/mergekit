# LRP Merge Method — Full Implementation Writeup
> **Project:** Mergekit Custom Merge Method  
> **Location:** `D:\Mergekit\mergekit\`  
> **Date Completed:** 27 March 2026  

---

## What Is This Project?

We implemented a brand-new **merge method** inside the open-source `mergekit` library called **LRP-Merge** (`merge_method: lrp`).

**Mergekit** is a library that lets you combine (merge) multiple LLMs into one model. It comes with built-in methods like:
- `linear` — simple weighted average of model weights
- `ties` — Task Vector merging with sign conflict resolution
- `dare_ties` — like TIES but with random delta pruning
- `slerp` — spherical linear interpolation

We added a custom method powered by **Layer-wise Relevance Propagation (LRP)** scores.

---

## The Core Idea — What Is LRP-Merge?

### Problem With Standard Merging
```
θ_merged = 0.7 × θ_model_A + 0.3 × θ_model_B
```
This treats every weight as equally important. In reality, only a small fraction of weights drive a model's learned capabilities. The rest are noise.

### The LRP Solution
**Layer-wise Relevance Propagation** is an XAI technique that assigns each weight a score: *"How much did this weight contribute to the correct prediction?"* The result is a **relevance map** — a tensor of the same shape as the weight matrix.

### Our 3-Step Algorithm

#### Step 1 — Compute Task Vector (Delta)
```
δ = θ_model - θ_base
```
We use the delta (deviation from base) rather than raw weights, so the base model's knowledge is always preserved.

#### Step 2 — Functional Trimming (Sparsification)
```
mask = (r >= threshold)    ← top k% most relevant entries
δ_sparse = δ × mask
```
- If a `.pt` LRP map exists for a weight → use it as scoring `r`
- If not → fall back to magnitude scoring: `r = |δ|`
- **NEVER sparsify 1D tensors** (biases, LayerNorm scales) — zeroing those destroys the layer

#### Step 3 — Weighted Parameter Averaging
```
θ_merged = θ_base + Σᵢ (λᵢ / Σλ) × δᵢ_sparse
```
The base model is preserved intact; we add the filtered knowledge on top.

Key behaviors:
- LRP maps loaded from `./lrp_scores/<model_name>/<weight_name>.pt`
- Graceful fallback to magnitude pruning if maps are missing
- `base_model_ref` passed into the task so base is identified correctly
- 1D tensor guard: `if len(delta.shape) >= 2` before any sparsification

#### [lrp_config.yaml](file:///d:/Mergekit/mergekit/lrp_config.yaml)
```yaml
merge_method: lrp
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
parameters:
  density: 0.7
models:
  - model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    parameters:
      weight: 0.7
  - model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    parameters:
      weight: 0.3
```



## Command Workflow (Run Order)

```powershell
# Step 1 — Install mergekit in editable mode (do this once)
pip install -e .

# Step 2 — Clear any old output before re-merging (Windows required)
Remove-Item -Recurse -Force ./merged-model-directory/*

# Step 3 — Run the merge engine
mergekit-yaml lrp_config.yaml ./merged-model-directory `
    --copy-tokenizer `
    --lazy-unpickle `
    --allow-crimes `
    --out-shard-size 300M

# Step 4 — Test inference on the merged model
python test_merge.py
```

---

## Final Architecture Diagram

```
lrp_config.yaml
      |
      v
mergekit-yaml CLI
      |
      v
plan.py  (builds the computation DAG)
      |
      v  for each tensor layer in the model:
LRPMerge.make_task()
      |
      v
LRPMergeTask.execute(tensors)
      |
      +-- Load base tensor
      |
      +-- For each non-base model:
      |       delta = model_tensor - base_tensor
      |       r = load ./lrp_scores/.../weight.pt  (or |delta| fallback)
      |       mask = top density% of r
      |       delta_sparse = delta * mask  (skip for 1D tensors)
      |
      +-- merged = base + Σ (λᵢ/Σλ × delta_sparseᵢ)
      |
      v
Saved to ./merged-model-directory/*.safetensors (8 shards)
      |
      v
test_merge.py → AutoModelForCausalLM → coherent text generation ✅
```

---

## Next Steps

1. **Generate real LRP maps**: Run a LRP attribution pass on your fine-tuned models using a dataset; save per-weight score tensors as `.pt` files.
2. **Place maps in**: `./lrp_scores/<model_name_with_slashes_replaced_by_underscores>/<weight.name>.pt`
3. **Re-run the merge**: The system will automatically detect and use them instead of the magnitude fallback.
4. **Compare outputs**: The LRP-guided model should be more capable at the tasks the relevance scores were computed for, versus the magnitude-pruned baseline.
