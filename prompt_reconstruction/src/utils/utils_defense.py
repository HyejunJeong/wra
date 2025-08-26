import random
import pandas as pd
import os

def load_decoy_domains(example, args):
    idx = example['Idx']
    
    # Calculate the base file number for this idx (1-20 maps to 1-100)
    # Each idx gets 5 consecutive files: idx*5-4, idx*5-3, idx*5-2, idx*5-1, idx*5
    base_file_num = idx * 5
    
    # Load 5 decoy domain files for this idx
    decoy_domains_list = []
    for i in range(5):
        file_num = base_file_num - 4 + i  # This gives us: base_file_num-4, base_file_num-3, base_file_num-2, base_file_num-1, base_file_num
        decoy_domains_file = os.path.join(args.Decoy_domains_folder, f"domains_{file_num:03d}.txt")
        
        try:
            with open(decoy_domains_file, 'r') as file:
                decoy_domains = file.read().splitlines()
                decoy_domains_list.append(decoy_domains)
        except FileNotFoundError:
            print(f"Warning: Could not find file {decoy_domains_file}")
            decoy_domains_list.append([])
    
    # Add 5 new columns to the example DataFrame
    for i, domains in enumerate(decoy_domains_list):
        col_name = f"decoy_domains_{i+1}"
        example[col_name] = domains
    
    return decoy_domains_list

def load_decoy_prompts(example, args):
    idx = example['Idx']
    
    # Calculate the base line number for this idx (1-20 maps to lines 1-100)
    # Each idx gets 5 consecutive lines: idx*5-4, idx*5-3, idx*5-2, idx*5-1, idx*5
    base_line_num = idx * 5
    
    # Path to the decoy prompts file
    decoy_prompts_file = os.path.join(args.domain_prompts_folder, f"decoy_prompts.txt")
    
    # Load 5 decoy prompts for this idx
    decoy_prompts_list = []
    try:
        with open(decoy_prompts_file, 'r') as file:
            all_lines = file.readlines()
            
            # Load 5 consecutive lines for this idx
            for i in range(5):
                line_num = base_line_num - 4 + i  # This gives us: base_line_num-4, base_line_num-3, base_line_num-2, base_line_num-1, base_line_num
                
                # Adjust for 0-based indexing (line_num - 1)
                if 0 <= (line_num - 1) < len(all_lines):
                    prompt = all_lines[line_num - 1].strip()
                    decoy_prompts_list.append(prompt)
                else:
                    print(f"Warning: Line {line_num} is out of range in {decoy_prompts_file}")
                    decoy_prompts_list.append("")
                    
    except FileNotFoundError:
        print(f"Warning: Could not find file {decoy_prompts_file}")
        decoy_prompts_list = [""] * 5
    
    # Add 5 new columns to the example DataFrame
    for i, prompt in enumerate(decoy_prompts_list):
        col_name = f"decoy_prompt_{i+1}"
        example[col_name] = prompt
    
    return decoy_prompts_list

def sample_traces_defence(example, args, sampling_ratio = 1.0):
    load_decoy_domains(example, args)
    load_decoy_prompts(example, args)
    
    traces_col_names = ['domain_sequence']
    # randomly select the decoy domains sets to use
    set_nums_to_use = random.sample(range(1, 6), args.decoy_domains_set_num)
    for set_num in set_nums_to_use:
        decoy_domains_col_name = f"decoy_domains_{set_num}"
        traces_col_names.append(decoy_domains_col_name)
    
    if args.mixing_sets_strategy == "random":
        random.shuffle(traces_col_names)
    
    
    all_traces = []
    # Process domain_sequence column
    for sequence_col in traces_col_names:
        sequence_value = example[sequence_col]
        if isinstance(sequence_value, str) and ',' in sequence_value:
            # Split the sequence and clean up
            items = [item.strip() for item in sequence_value.split(',') if item.strip()]
            all_traces.extend(items)
        elif isinstance(sequence_value, list):
            all_traces.extend(sequence_value)
        elif sequence_value:
            all_traces.append(str(sequence_value))
    
   
    
    # Calculate how many items to keep based on sampling ratio
    num_to_keep = max(1, int(len(all_traces) * sampling_ratio))
    all_traces = all_traces[:num_to_keep]
    # Mixing the traces
    if args.mixing_traces_strategy == "random":
        random.shuffle(all_traces)
    
    return all_traces
