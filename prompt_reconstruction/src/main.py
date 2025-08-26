import os
import json
import argparse
from utils.utils import load_data, get_data_embeddings, Run_ICL_Loop, Save_CSV_Data
from utils.utils_model import get_llm_model


def main():
    parser = argparse.ArgumentParser(description="Prompt reconstruction using ICL.")
    parser.add_argument("--config", type=str, default="configs/config.json", help="Path to JSON config file.")
    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Parse config as arguments
    for key, value in config.items():
        setattr(args, key, value)
    # Set random seed for reproducibility
    #set_seed(args.seed if hasattr(args, "seed") else 0)

    # Load training data
    whole_train_data, train_filenames = load_data(args.traning_data_folder)
    whole_test_data, test_filenames = load_data(args.test_data_folder)

    #Get embeddings of the data based on the domain
    print("Generating embeddings for training data...")
    train_data_embeddings = get_data_embeddings(whole_train_data, args)
    
    print("Generating embeddings for test data...")
    test_data_embeddings = get_data_embeddings(whole_test_data, args)

    llm_model = get_llm_model(args)

    # Handle ICL Examples
    reconstructed_test_data = Run_ICL_Loop(llm_model, whole_train_data, whole_test_data, train_data_embeddings, test_data_embeddings, args)
    
    Save_CSV_Data(reconstructed_test_data, test_filenames, args)
   
   
if __name__ == "__main__":
    main()