#!/usr/bin/env python3

import os
import shutil
import subprocess
import argparse

def validate(args):
    print("Step 1: Checking local model paths...")
    for path in [args.model1, args.model2]:
        if os.path.exists(path):
            print(f"Found: {path}")
        else:
            print(f" Warning: Missing model: {path}")

def generate_yaml(args):
    print("Step 2: Generating YAML...")
    
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
      lrp_scores: \"{args.model1}/lrp_scores.safetensors\"

  - model: \"{args.model2}\"
    parameters:
      weight: 1.0
      lrp_scores: \"{args.model2}/lrp_scores.safetensors\"
"""
    with open("lrp_config.yaml", "w") as f:
        f.write(yaml.strip())

    print("\nYAML Generated:\n")
    print(yaml)

def compute_lrp_scores(args):
    print("\nStep 2.5: Computing LRP scores (if requested)...\n")
    if not args.compute_lrp:
        print("Skipping LRP computation. Make sure you already have lrp_scores.safetensors in your model directories.")
        return
        
    print("Computing LRP scores for Model 1...")
    subprocess.run([
        "python", "lrp_computer.py", args.model1, args.model1,
        "--prompts", "The capital of France is"
    ], check=True)
    
    print("Computing LRP scores for Model 2...")
    subprocess.run([
        "python", "lrp_computer.py", args.model2, args.model2,
        "--prompts", "The capital of France is"
    ], check=True)

def run_merge(args):
    print("\nStep 3: Running merge...\n")
    mergekit_exec = "/usr/local/bin/mergekit-yaml"
    if not os.path.exists(mergekit_exec):
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
    parser.add_argument("--cuda", action="store_true", default=True, help="Use CUDA for merging")
    args = parser.parse_args()

    print("=== LRP MERGE PIPELINE START ===\n")
    validate(args)
    compute_lrp_scores(args)
    generate_yaml(args)
    os.makedirs(args.output, exist_ok=True)
    run_merge(args)
    print("\n✓ ALL DONE")

if __name__ == "__main__":
    main()
