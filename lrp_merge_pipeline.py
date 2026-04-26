#!/usr/bin/env python3

import os
import sys
import shutil
import subprocess
import argparse

def validate(args):
    """Validate that all required model paths exist. Fail fast with clear error."""
    print("Step 1: Checking local model paths...")
    missing_paths = []
    
    for path in [args.model1, args.model2]:
        if os.path.exists(path):
            print(f"✓ Found: {path}")
        else:
            print(f"✗ Missing: {path}")
            missing_paths.append(path)
    
    if missing_paths:
        raise FileNotFoundError(
            f"Required model paths not found:\n" +
            "\n".join(f"  - {p}" for p in missing_paths) +
            "\n\nPlease ensure models exist or provide correct paths via --model1 and --model2"
        )

def generate_yaml(args):
    print("Step 2: Generating YAML...")
    
    m1_score_path = os.path.join(args.output, "lrp_scores", "model1", "lrp_scores.safetensors") if args.compute_lrp else f"{args.model1}/lrp_scores.safetensors"
    m2_score_path = os.path.join(args.output, "lrp_scores", "model2", "lrp_scores.safetensors") if args.compute_lrp else f"{args.model2}/lrp_scores.safetensors"
    
    yaml = f"""
merge_method: lrp

base_model:
  model: \"{args.base_model}\"

parameters:
  density: 0.7

models:
  - model: \"{args.model1}\"
    parameters:
      weight: 1.0
      lrp_scores: \"{m1_score_path}\"

  - model: \"{args.model2}\"
    parameters:
      weight: 1.0
      lrp_scores: \"{m2_score_path}\"
"""
    with open("lrp_config.yaml", "w") as f:
        f.write(yaml.strip())

    print("\nYAML Generated:\n")
    print(yaml)

def compute_lrp_scores(args):
    """Compute LRP scores for both models using the correct script path."""
    print("\nStep 2.5: Computing LRP scores (if requested)...\n")
    if not args.compute_lrp:
        print("Skipping LRP computation. Make sure you already have lrp_scores.safetensors in your model directories.")
        return
    
    # Resolve lrp_computer.py relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lrp_computer_path = os.path.join(script_dir, "lrp_computer.py")
    
    if not os.path.exists(lrp_computer_path):
        raise FileNotFoundError(
            f"lrp_computer.py not found at {lrp_computer_path}\n"
            f"Expected it next to {__file__}"
        )
        
    m1_out = os.path.join(args.output, "lrp_scores", "model1")
    m2_out = os.path.join(args.output, "lrp_scores", "model2")
    os.makedirs(m1_out, exist_ok=True)
    os.makedirs(m2_out, exist_ok=True)
        
    print("Computing LRP scores for Model 1...")
    subprocess.run([
        sys.executable, lrp_computer_path, args.model1, m1_out,
        "--prompts", "The capital of France is"
    ], check=True)
    
    print("Computing LRP scores for Model 2...")
    subprocess.run([
        sys.executable, lrp_computer_path, args.model2, m2_out,
        "--prompts", "The capital of France is"
    ], check=True)

def run_merge(args):
    print("\nStep 3: Running merge...\n")
    mergekit_exec = shutil.which("mergekit-yaml")
    if not mergekit_exec:
        raise RuntimeError("✗ mergekit-yaml not found. Install mergekit.")

    cmd = [
        mergekit_exec,
        "lrp_config.yaml",
        args.output,
        "--copy-tokenizer",
        "--allow-crimes",
    ]
    if args.cuda:
        cmd.append("--cuda")

    print("Running command:")
    print(" ".join(cmd), "\n")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"✗ Merge failed with code {res.returncode}")
    print("\nMerge completed successfully!")
    print(f"📁 Output: {args.output}")

def main():
    parser = argparse.ArgumentParser(description="LRP Merge Pipeline")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model path or name")
    parser.add_argument("--model1", default="./models/tinyllama-global-full", help="First model path")
    parser.add_argument("--model2", default="./models/tinyllama-local-full", help="Second model path")
    parser.add_argument("--output", default="./models/merged-model", help="Output directory")
    parser.add_argument("--compute-lrp", action="store_true", help="Compute LRP scores before merging")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device to use for merging")
    args = parser.parse_args()

    args.cuda = args.device == "cuda"

    print("=== LRP MERGE PIPELINE START ===\n")
    validate(args)
    compute_lrp_scores(args)
    generate_yaml(args)
    os.makedirs(args.output, exist_ok=True)
    run_merge(args)
    print("\n✓ ALL DONE")

if __name__ == "__main__":
    main()
