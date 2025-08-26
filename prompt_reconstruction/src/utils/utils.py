import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.utils_ICL_prompt import Choose_ICL_Examples, Build_ICL_Prompt
from utils.utils_model import get_llm_model, get_llm_response

def load_data(data_folder):
    data = []
    filenames = []
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(data_folder, file)
            df = pd.read_csv(file_path)
            data.append(df)
            filenames.append(file)
    return data, filenames

def get_data_embeddings(whole_data, args):
    """
    Get embeddings for data based on domain_sequence or url_sequence column using SBERT.
    
    Args:
        whole_data: List of DataFrames containing the data
        args: Arguments object containing configuration and icl_trace_type
        
    Returns:
        List of embeddings for each dataset
    """
    print("Loading SBERT model for similarity-based selection...")
    sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder="work/pi_ahoumansadr_umass_edu/mteymoorianf/Research/Web_Agent/Models_Cache")
    
    
    all_embeddings = []
    
    for dataset_index, dataset in enumerate(whole_data):
        print(f"Processing dataset {dataset_index + 1}/{len(whole_data)}")
        
        # Choose the appropriate column based on trace type
        if args.icl_trace_type == "domain":
            sequence_col = 'domain_sequence'
        elif args.icl_trace_type == "URL":
            sequence_col = 'url_sequence'
        else:
            raise ValueError(f"Invalid ICL trace type: {args.icl_trace_type}. Supported types: 'domain', 'URL'")
        
        # Get sequence text for embedding
        sequence_texts = []
        for idx, row in dataset.iterrows():
            sequence_value = row[sequence_col]
            
            # Handle different formats of sequence data
            if isinstance(sequence_value, str):
                # If it's a comma-separated string, take the first few items
                if ',' in sequence_value:
                    items = sequence_value.split(',')[:] 
                    sequence_text = ' '.join([item.strip() for item in items])
                else:
                    sequence_text = sequence_value
            else:
                # If it's not a string, convert to string
                sequence_text = str(sequence_value)
            
            sequence_texts.append(sequence_text)
        
        # Generate embeddings
        try:
            embeddings = sbert_model.encode(sequence_texts, convert_to_tensor=True)
            all_embeddings.append(embeddings)
            print(f"Generated {len(embeddings)} embeddings for dataset {dataset_index}")
        except Exception as e:
            print(f"Error generating embeddings for dataset {dataset_index}: {e}")
            all_embeddings.append([])
    
    return all_embeddings

def Save_CSV_Data(reconstructed_test_data, test_filenames, args):
    # Save reconstructed test data
    if args.icl_type == "contrastive":
        subfoler_output_name = args.model + "_" + args.icl_type + "_" + args.icl_contrastive_strategy + "_" + str(args.icl_contrastive_num_negatives) + "_" + args.icl_trace_type + "_" + str(args.icl_num_examples) + "_" + args.icl_selection_strategy + "_" + args.icl_ordering_strategy
    elif args.timing_info:
        subfoler_output_name = "timing_info_" + args.model + "_" + args.icl_type + "_" + args.icl_trace_type + "_" + str(args.sampling_ratio) + "_" + str(args.icl_num_examples) + "_" + args.icl_selection_strategy + "_" + args.icl_ordering_strategy
    elif args.defence:
        subfoler_output_name = "defence_" + str(args.decoy_domains_set_num) + "_" + args.mixing_sets_strategy + "_" + args.mixing_traces_strategy
    else:
        subfoler_output_name = args.model + "_" + args.icl_type + "_" + args.icl_trace_type + "_" + str(args.sampling_ratio) + "_" + str(args.icl_num_examples) + "_" + args.icl_selection_strategy + "_" + args.icl_ordering_strategy
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir + "/" + subfoler_output_name, exist_ok=True)
    
    for test_data_index, test_data in enumerate(reconstructed_test_data):
        # Use original filename with reconstructed prefix
        original_filename = test_filenames[test_data_index]
        output_name = "reconstructed_" + original_filename
        test_data.to_csv(args.output_dir + "/" + subfoler_output_name + "/" + output_name, index=False)


def Run_ICL_Loop(llm_model,whole_train_data, whole_test_data, train_data_embeddings, test_data_embeddings, args):
    # Add initial delay for Gemini models to help with rate limiting
    if args.model == "gemini-2.5-pro":
        import time
        print("Adding initial delay for Gemini API rate limiting...")
        time.sleep(5)  # Wait 5 seconds before starting
    
    # Run ICL Loop
    for test_data_index, test_data in enumerate(whole_test_data):
        print(f"Processing test data {test_data_index + 1}/{len(whole_test_data)}")
        for single_data_index, single_data in enumerate(test_data.iterrows()):
            single_data_embeddings = test_data_embeddings[test_data_index][single_data_index]
            icl_examples = Choose_ICL_Examples(whole_train_data[test_data_index], train_data_embeddings[test_data_index], single_data_embeddings, args)
            icl_prompt = Build_ICL_Prompt(icl_examples, single_data, llm_model,args)
            reconstructed_prompt = get_llm_response(llm_model, icl_prompt, args)
            test_data.loc[single_data_index, 'reconstructed_prompt'] = reconstructed_prompt
            if args.sampling_ratio < 1.0:
                col_name = "domain_" + str(args.sampling_ratio) + "_visibility"
                test_data.loc[single_data_index, col_name] = ', '.join(single_data[1][col_name])
            
            # Add delay between API calls to help with rate limiting
            if args.model == "gemini-2.5-pro":
                import time
                time.sleep(2)  # Wait 2 seconds between calls

    return whole_test_data