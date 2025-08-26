from sentence_transformers import SentenceTransformer
from utils.utils_model import get_llm_model
import argparse
import json
import os
from utils.utils_eval import *

def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--config", type=str, default="configs/config_eval.json", help="Path to JSON config file.")
    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Parse config as arguments
    for key, value in config.items():
        setattr(args, key, value)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    llm_model = get_llm_model(args)

    print(f"Loading SBERT model: {args.sbert_model}")
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder="work/pi_ahoumansadr_umass_edu/mteymoorianf/Research/Web_Agent/Models_Cache")

    # Determine which files to process
    if args.input_folder:
        # Use the HandleWholeFolder function to process all files
        HandleWholeFolder(args.input_folder, llm_model, sbert_model, args)
    else:
        print("No input folder provided.")

if __name__ == "__main__":
    main() 