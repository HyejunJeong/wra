import numpy as np
import random
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils_model import get_llm_response
from utils.utils_defence import sample_traces_defence

sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder="work/pi_ahoumansadr_umass_edu/mteymoorianf/Research/Web_Agent/Models_Cache")

def Choose_Embedding_ICL_Examples(train_data, train_data_embeddings, single_data_embeddings, args):
    """
    Choose ICL examples based on cosine similarity between single test data and training data embeddings.
    
    Args:
        single_data: Single test data point (row from test dataset)
        train_data: DataFrame containing training data
        train_data_embeddings: Pre-computed embeddings for training data
        single_data_embeddings: Pre-computed embedding for the single test data point
        args: Arguments object containing configuration
        dataset_index: Index of the current dataset in the embeddings list
        
    Returns:
        DataFrame with selected ICL examples
    """
    
    # Convert train embeddings to numpy array if it's a tensor
    if hasattr(train_data_embeddings, 'cpu'):
        train_embeddings = train_data_embeddings.cpu().numpy()
    elif hasattr(train_data_embeddings, 'numpy'):
        train_embeddings = train_data_embeddings.numpy()
    
    # Convert single data embedding to numpy array if it's a tensor
    if hasattr(single_data_embeddings, 'cpu'):
        single_embedding = single_data_embeddings.cpu().numpy()
    elif hasattr(single_data_embeddings, 'numpy'):
        single_embedding = single_data_embeddings.numpy()
    else:
        single_embedding = single_data_embeddings
    
    # Reshape single embedding to 2D array for cosine_similarity
    if single_embedding.ndim == 1:
        single_embedding = single_embedding.reshape(1, -1)
    
    # Calculate cosine similarity between single test data and all training examples
    similarities = cosine_similarity(single_embedding, train_embeddings)[0]
    
    # Select examples with highest similarity to the test data (most similar)
    # Get indices of examples with highest similarity scores
    selected_indices = np.argsort(similarities)[::-1][:args.icl_num_examples]
    
    # Return selected examples
    return train_data.iloc[selected_indices].reset_index(drop=True), selected_indices

def Ordering_ICL_Examples(icl_examples, selected_indices, args):
    if args.icl_ordering_strategy == "random":
        # Shuffle the examples randomly
        return icl_examples.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    elif args.icl_ordering_strategy == "ascending":
        # Reverse the order of ICL examples
        return icl_examples.iloc[::-1].reset_index(drop=True)
    elif args.icl_ordering_strategy == "descending":
        return icl_examples
    else:
        raise ValueError(f"Invalid ICL ordering strategy: {args.icl_ordering_strategy}. Supported types: 'random', 'ascending', 'descending'")
   
def Choose_ICL_Examples(train_data, train_data_embeddings, single_data_embeddings, args):
    if args.icl_selection_strategy == "random":
        return train_data.sample(n=args.icl_num_examples, random_state=args.seed)
    elif args.icl_selection_strategy == "embedding":
        icl_examples, selected_indices = Choose_Embedding_ICL_Examples(train_data, train_data_embeddings, single_data_embeddings, args)
        return Ordering_ICL_Examples(icl_examples, selected_indices, args)
    else:
        raise ValueError(f"Invalid ICL type: {args.icl_type}. Supported types: 'random', 'embedding'")
   
def sample_traces(example, args, sampling_ratio = 1.0):
    """
    Sample traces (domains or URLs) from an example based on trace type and sampling ratio.
    
    Args:
        example: DataFrame row containing domain_sequence and/or url_sequence
        args: Arguments object containing icl_trace_type and sampling_ratio
        
    Returns:
        List of sampled traces (domains or URLs)
    """
    if args.icl_trace_type == "domain":
        sequence_col = 'domain_sequence'
    elif args.icl_trace_type == "URL":
        sequence_col = 'url_sequence'
    else:
        raise ValueError(f"Invalid ICL trace type: {args.icl_trace_type}. Supported types: 'domain', 'URL', 'defence'")
    
    if sequence_col not in example:
        return []
    
    sequence_value = example[sequence_col]
    
    if not isinstance(sequence_value, str) or ',' not in sequence_value:
        return [sequence_value] if sequence_value else []
    
    # Split the sequence and clean up
    items = [item.strip() for item in sequence_value.split(',') if item.strip()]
    
    if not items:
        return []
    
    # Calculate how many items to keep based on sampling ratio
    num_to_keep = max(1, int(len(items) * sampling_ratio))
    
    # Randomly sample num_to_keep items from the list
    sampled_items = random.sample(items, num_to_keep)

    if sampling_ratio < 1.0:
        example['domain_' + str(sampling_ratio) + "_visibility"] = sampled_items
        
    return sampled_items

def get_normal_ICL_Prompt(icl_examples, single_data, args):
    # Determine trace type for prompt text
    if args.icl_trace_type == "domain":
        trace_label = "domains"
        trace_description = "domains visited by a web agent during a browsing session"
    elif args.icl_trace_type == "URL":
        trace_label = "URLs"
        trace_description = "URLs visited by a web agent during a browsing session"
    else:
        raise ValueError(f"Invalid ICL trace type: {args.icl_trace_type}. Supported types: 'domain', 'URL'")
    
    prompt = (
        "Objective:\n"
        f"You are given a list of {trace_description}.\n"
        "Your task is to reconstruct the most plausible **original user prompt** that would have caused\n"
        f"the agent to visit these {trace_label}.\n"
        "This prompt should reflect the likely information need or task of the user, as inferred from\n"
        f"the {trace_label}.\n"
    )
    if args.icl_num_examples > 0:
        prompt += "Here are some examples:\n\n"
    if args.timing_info:
        prompt += "Note: The timing information is provided in the format of seconds since the start of the session, showing the time the agent visited each domain. The first time is always 0. It has been shown in parenthesis after each domain.\n"
    
    for i, example in enumerate(icl_examples.iterrows(), 1):
        _, example_row = example
        traces = sample_traces(example_row, args, sampling_ratio = 1.0)
        
        # Handle case where 'prompts' column might not exist
        if 'prompts' in example_row:
            query = example_row['prompts']
        else:
            query = "Original prompt not available"
            
        prompt += f"Example {i}:\n"
        prompt += f"Visited {trace_label.capitalize()}:\n"
        for trace in traces:
            prompt += f"- {trace}\n"
        prompt += f"\nReconstructed Prompt:\n\"{query}\"\n\n"
    
    prompt += (
        f"Now, for the following set of visited {trace_label}, reconstruct the most plausible original user prompt.\n\n"
        "Important: Only output the reconstructed prompt itself. Do not output the visited {trace_label}, reasoning, or any other text.\n\n"
        f"Visited {trace_label.capitalize()}:\n"
    )
    
    # Sample traces for the single_data as well
    if args.defence:
        single_traces = sample_traces_defence(single_data[1], args, sampling_ratio = args.sampling_ratio)  # single_data is a tuple (index, row)
    else:
        single_traces = sample_traces(single_data[1], args, sampling_ratio = args.sampling_ratio)  # single_data is a tuple (index, row)
    for trace in single_traces:
        prompt += f"- {trace}\n"
    prompt += "\nReconstructed Prompt:\n"
    return prompt

def get_less_preferred_queries(traces, preferred_query, num_negatives, llm_model, args):
    prompt = f"""
            You are helping create contrastive examples for in-context learning. Given visited web domains and a preferred user query, generate {num_negatives} *less preferred* queries.

            The less preferred queries should still be somewhat related to the domains, but should be:
            - vague, or
            - off-topic, or
            - missing critical details

            Each query should be different from the others and represent different types of "bad" queries.

            ---

            Visited Domains:
            {', '.join(traces)}

            Preferred Query:
            "{preferred_query}"

            Generate {num_negatives} less preferred queries:

            Note: Only output the less preferred queries, do not output any other text.
            Output format: on each line, output the less preferred query. not numbering.

            """
    
    response = get_llm_response(llm_model, prompt, args)
    
    # Parse the response into a list of queries
    neg_queries = []
    lines = response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue    
        neg_queries.append(line)
    
    if args.icl_contrastive_strategy == "simple":
        return neg_queries
    elif args.icl_contrastive_strategy == "quality_filtered":
        emb1 = sbert_model.encode(preferred_query, convert_to_tensor=True)
        
        # Keep generating new queries until we have enough that are below threshold
        final_neg_queries = []
        max_attempts = num_negatives * 10  # Limit attempts to avoid infinite loops
        attempts = 0
        
        while len(final_neg_queries) < num_negatives and attempts < max_attempts:
            # Check existing queries first
            for neg_query in neg_queries:
                if len(final_neg_queries) >= num_negatives:
                    break
                    
                emb2 = sbert_model.encode(neg_query, convert_to_tensor=True)
                sbert_sim = float(util.pytorch_cos_sim(emb1, emb2)[0][0])
                
                if sbert_sim < args.icl_contrastive_quality_threshold:
                    final_neg_queries.append(neg_query)
            
            # If we still need more queries, generate new ones
            if len(final_neg_queries) < num_negatives:
                # Generate additional queries
                additional_prompt = f"""
                Generate {num_negatives - len(final_neg_queries)} additional less preferred queries for the same context.
                
                Visited Domains:
                {', '.join(traces)}
                
                Preferred Query:
                "{preferred_query}"
                
                Already accepted negative queries:
                {chr(10).join([f"- {q}" for q in final_neg_queries])}
                
                Generate {num_negatives - len(final_neg_queries)} new less preferred queries that are:
                - vague, or
                - off-topic, or  
                - missing critical details
                - different from the already accepted ones
                
                Output format: on each line, output the less preferred query. not numbering.
                """
                
                additional_response = get_llm_response(llm_model, additional_prompt, args)
                additional_queries = []
                for line in additional_response.strip().split('\n'):
                    line = line.strip()
                    if line:
                        additional_queries.append(line)
                
                # Replace the original neg_queries with new ones for next iteration
                neg_queries = additional_queries
                attempts += 1
            else:
                break
        
        # If we still don't have enough after max attempts, return what we have
        if len(final_neg_queries) < num_negatives:
            print(f"Warning: Could only generate {len(final_neg_queries)} quality-filtered negative queries out of {num_negatives} requested")
        
        return final_neg_queries
       
    else:
        raise ValueError(f"Invalid ICL contrastive strategy: {args.icl_contrastive_strategy}. Supported types: 'simple', 'multiple'")

def get_contrastive_ICL_Prompt(icl_examples, single_data, llm_model, args):
    # Determine trace type for prompt text
    if args.icl_trace_type == "domain":
        trace_label = "domains"
        trace_description = "domains visited by a web agent during a browsing session"
    elif args.icl_trace_type == "URL":
        trace_label = "URLs"
        trace_description = "URLs visited by a web agent during a browsing session"
    else:
        raise ValueError(f"Invalid ICL trace type: {args.icl_trace_type}. Supported types: 'domain', 'URL'")
    
    prompt = (
        "Objective:\n"
        f"You are given a list of {trace_description}.\n"
        "Your task is to suggest a good, preferred prompt (query) for the information need that best matches the set of visited web domains provided.\n"
        f"You will see examples with visited {trace_label}, a preferred (specific) query, multiple less preferred (vague or off-topic) queries, and reasoning.\n"
        f"The preferred query should be specific, actionable, and match the {trace_label} content.\n"
        "The less preferred queries should be vague, off-topic, or missing critical details.\n"
        "Here are some examples:\n\n"
    )
    for i, example in enumerate(icl_examples.iterrows(), 1):
        _, example_row = example
        traces = sample_traces(example_row, args, sampling_ratio = 1.0)
        
        # Handle case where 'prompts' column might not exist
        if 'prompts' in example_row:
            preferred_query = example_row['prompts']
        else:
            preferred_query = "Original prompt not available"
            
        less_preferred_queries = get_less_preferred_queries(traces, preferred_query, args.icl_contrastive_num_negatives, llm_model, args)
        
        prompt += f"Example {i}:\n"
        prompt += f"Visited {trace_label.capitalize()}:\n"
        for trace in traces:
            prompt += f"- {trace}\n"
        prompt += f"\nPreferred Query:\n\"{preferred_query}\"\n\n"
        
        # Include all negative examples
        if len(less_preferred_queries) == 1:
            prompt += f"Less Preferred Query:\n\"{less_preferred_queries[0]}\"\n\n"
        else:
            prompt += "Less Preferred Queries:\n"
            for j, neg_query in enumerate(less_preferred_queries, 1):
                prompt += f"{j}. \"{neg_query}\"\n"
            prompt += "\n"
        
        prompt += (
            "Reasoning:\n"
            f"The preferred query is specific, actionable, and matches {trace_label} content. "
            "The less preferred queries are vague, off-topic, or missing critical details.\n\n"
        )

    prompt += (
        f"Now, for the following set of visited {trace_label}, suggest a good, preferred prompt (query) that best matches the information need implied by these domains.\n\n"
        "Important: Only output the preferred query itself. Do not output the less preferred query, reasoning, or any other text.\n\n"
        f"Visited {trace_label}:\n"
    )
    # Sample traces for the single_data as well
    single_traces = sample_traces(single_data[1], args, sampling_ratio = args.sampling_ratio)  # single_data is a tuple (index, row)
    for trace in single_traces:
        prompt += f"- {trace}\n"
    prompt += "\nPreferred Query:\n"
    return prompt

def Build_ICL_Prompt(icl_examples, single_data, llm_model, args):
    if args.icl_type == "normal":
        prompt = get_normal_ICL_Prompt(icl_examples, single_data, args)
    elif args.icl_type == "contrastive":
        prompt = get_contrastive_ICL_Prompt(icl_examples, single_data, llm_model, args)
    else:
        raise ValueError(f"Invalid ICL type: {args.icl_type}. Supported types: 'normal', 'contrastive'")
    return prompt
