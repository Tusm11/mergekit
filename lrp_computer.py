"""
LRP (Layer-wise Relevance Propagation) Score Computation Module.
This module provides functionality to compute LRP relevance scores for model weights,
which can then be used by the LRP-Merge method for intelligent model merging.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LRPConfig:
    """Configuration for LRP computation."""

    model_path: str
    output_path: str
    sample_prompts: List[str]
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LRPComputer:
    """
    Computes Layer-wise Relevance Propagation scores for transformer models using AttnLRP.
    """

    def __init__(self, config: LRPConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.relevance_scores: Dict[str, torch.Tensor] = {}

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        print(f"Loading model from {self.config.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        # Determine dtype based on device
        has_cuda = torch.cuda.is_available() and self.config.device == "cuda"
        if has_cuda:
            torch_dtype = torch.float16
            device_map = self.config.device
        else:
            torch_dtype = torch.float32
            device_map = None  # device_map not recommended for CPU

        print(f"  Using device: {self.config.device}")
        print(f"  Using dtype: {torch_dtype}")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"  Failed with default settings: {e}")
            print("  Trying with trust_remote_code=True...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        # Explicitly move to CPU if needed
        if self.config.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()

    def compute_all_relevance_scores(self) -> Dict[str, torch.Tensor]:
        """
        Compute relevance scores for all model weights using true AttnLRP.
        """
        if self.model is None:
            self.load_model()

        try:
            model_type = getattr(self.model.config, "model_type", "").lower()
            if "llama" in model_type:
                from lxt.models.llama import attnlrp
                attnlrp.register(self.model)
            elif "qwen" in model_type:
                from lxt.models.qwen2 import attnlrp
                attnlrp.register(self.model)
            elif "mistral" in model_type:
                from lxt.models.mistral import attnlrp
                attnlrp.register(self.model)
            else:
                raise ValueError(
                    f"AttnLRP not supported for model_type={model_type!r}. "
                    f"Currently supported: llama, qwen, mistral. "
                    f"For other architectures, contribute an lxt rules module."
                )
        except ImportError:
            raise ImportError("lxt is required for AttnLRP. Install with: pip install lxt") from None

        # Free intermediate activations during backward — recompute instead of store
        self.model.gradient_checkpointing_enable()
        orig_use_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = False  # required with checkpointing

        try:
            # Accumulator on CPU; only the active sample's grads live on GPU
            relevance_acc = {
                n: torch.zeros_like(p, device="cpu", dtype=torch.float32)
                for n, p in self.model.named_parameters() if p.requires_grad
            }

            # Tokenize sample prompts
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            if self.config.sample_prompts:
                inputs = self.tokenizer(
                    self.config.sample_prompts, return_tensors="pt",
                    padding=True, truncation=True, max_length=self.config.max_length,
                ).to(self.config.device)
            else:
                raise ValueError("No sample prompts provided for LRP computation")

            n_samples = inputs["input_ids"].shape[0]

            for i in range(n_samples):                  # one sample at a time
                print(f"Processing sample {i+1}/{n_samples}...")
                ids = inputs["input_ids"][i:i+1]
                attention_mask = inputs["attention_mask"][i:i+1]
                
                logits = self.model(input_ids=ids, attention_mask=attention_mask).logits            # full graph, checkpointed
                
                # Find the actual last token position before padding (handles both left and right padding)
                last_token_idx = attention_mask[0].nonzero(as_tuple=True)[0][-1].item()
                target = logits[:, last_token_idx, :].max(dim=-1).values.sum()         # seed: predicted-token logit

                self.model.zero_grad(set_to_none=True)
                target.backward()                                          # ONE real backward pass

                with torch.no_grad():                                      # accumulate R_w = grad ⊙ w
                    for n, p in self.model.named_parameters():
                        if p.grad is None:
                            continue
                        relevance_acc[n] += (p.grad.detach() * p.detach()).abs().float().cpu()
                
                self.model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                
        finally:
            # Restore original settings
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = orig_use_cache

        self.relevance_scores = {n: (r / n_samples) for n, r in relevance_acc.items()}
        
        # Handle tied embeddings: copy embed_tokens score to lm_head if missing
        if getattr(self.model.config, "tie_word_embeddings", False):
            if "model.embed_tokens.weight" in self.relevance_scores and "lm_head.weight" not in self.relevance_scores:
                self.relevance_scores["lm_head.weight"] = self.relevance_scores["model.embed_tokens.weight"]
                
        return self.relevance_scores

    def save_relevance_scores(self, output_format: str = "safetensors") -> None:
        """Save computed relevance scores to disk."""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if output_format == "safetensors":
            try:
                from safetensors.torch import save_file

                save_file(self.relevance_scores, output_path / "lrp_scores.safetensors")
            except ImportError:
                print("safetensors not available, using torch.save")
                torch.save(self.relevance_scores, output_path / "lrp_scores.pt")
        else:
            torch.save(self.relevance_scores, output_path / "lrp_scores.pt")

        # Save metadata
        metadata = {
            "model_path": self.config.model_path,
            "num_tensors": len(self.relevance_scores),
        }

        with open(output_path / "lrp_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"LRP scores saved to {output_path}")


def compute_lrp_for_model(
    model_path: str,
    output_path: str,
    sample_prompts: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to compute LRP scores for a model.

    Args:
        model_path: Path to the HuggingFace model
        output_path: Where to save the LRP scores
        sample_prompts: List of sample prompts for LRP computation
        **kwargs: Additional configuration options

    Returns:
        Dictionary mapping tensor names to relevance scores
    """
    default_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "The capital of France is Paris.",
    ]

    config = LRPConfig(
        model_path=model_path,
        output_path=output_path,
        sample_prompts=sample_prompts or default_prompts,
        **kwargs,
    )

    computer = LRPComputer(config)
    scores = computer.compute_all_relevance_scores()
    computer.save_relevance_scores()

    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute LRP scores for a model")
    parser.add_argument("model_path", help="Path to the HuggingFace model")
    parser.add_argument("output_path", help="Where to save LRP scores")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--prompts", nargs="+", help="Sample prompts for LRP computation"
    )

    args = parser.parse_args()

    compute_lrp_for_model(
        model_path=args.model_path,
        output_path=args.output_path,
        sample_prompts=args.prompts,
        device=args.device,
    )
