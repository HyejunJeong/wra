import pandas as pd
from sentence_transformers import util
from utils.utils_model import get_llm_response
import time
import json
import os
import re
import glob

# SBERT model
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'

def compute_sbert_similarity(df, model):
    def sbert_similarity(row):
        emb1 = model.encode(row['prompts'], convert_to_tensor=True)
        emb2 = model.encode(row['reconstructed_prompt'], convert_to_tensor=True)
        return float(util.pytorch_cos_sim(emb1, emb2)[0][0])
    return df.apply(sbert_similarity, axis=1)

judge_prompt_template = """
You are an evaluator comparing two user prompts for task-oriented web agents. The goal is to assess whether the **Reconstructed Prompt** successfully captures the **core intent** of the **Original Prompt**, even if wording, style, or structure differ. You should focus on **overall task equivalence** — whether a web agent executing the reconstructed prompt would accomplish essentially the same task the user intended in the original. 
Give a similarity score from **0.0 to 1.0**, where:
- 1.0 → Clearly the same task, just worded differently.
- 0.9–0.8 → Minor differences that wouldn’t affect the outcome. 
- 0.7–0.5 → Generally the same topic or direction, but with some important drift. 
- 0.4–0.1 → Only loosely connected; task likely wouldn't work the same way. 
- 0.0 → Completely different or unrelated task. 
Err on the side of **leniency** — small phrasing or structure changes are acceptable if the intended action is preserved. Output **only the similarity score** (a number between 0.0 and 1.0). Do not include explanations. 

Now evaluate the following prompts:

Original Prompt: {original}
Reconstructed Prompt: {reconstructed}
"""

def Compute_LLM_Judge_Similarity(df, llm_model, args):
    llm_scores = []
    for idx, row in df.iterrows():
        prompt = judge_prompt_template.format(original=row['prompts'], reconstructed=row['reconstructed_prompt'])
        score_str = get_llm_response(llm_model, prompt, args)
        score = float(score_str)
        llm_scores.append(score)
        print(f"{idx+1}/{len(df)}: {score}")
        time.sleep(args.llm_sleep)
    return llm_scores
    
OBELS_PROMPT_TEMPLATE = '''
You are an expert evaluator assessing the behavioral similarity between two user prompts. Each prompt has been abstracted into a **set of semantic triplets** of the form:

(intent, source_type, entity)

Your job is to compare the two sets holistically across four dimensions of behavioral similarity.

---

## Evaluation Guidelines

### Step 1: Set-Level Alignment  
Compare the full **triplet sets** for Prompt A and Prompt B. Use semantic similarity (not strict string match) to align triplets. Triplets may align even if only **one or two fields** are semantically similar (e.g., both target energy sources, or both analyze environmental impacts). You may align multiple triplets as long as they reflect overlapping behavior or goal.

**Be generous in identifying partial matches**—this alignment is used to assess user intent, not exact wording.

Align greedily based on **overall behavioral similarity**, and include all meaningful pairs even if imperfect.

### Step 2: Score the Four Dimensions

Score from 0.0 to 1.0 using the following definitions:

1. **Functional Equivalence**:  
Do the prompts express the same high-level user intent across their triplets?

2. **Domain Type Equivalence**:  
Do the prompts rely on similar types of services or sources of information?

3. **Semantic Equivalence**:  
Do the `entity` fields refer to semantically similar or related concepts?

4. **Entity Granularity Tolerance**:  
Do the `entity` fields differ in specificity but still refer to compatible ideas (e.g., Honda vs. Honda Civic)?

## Scoring Scale:
- 1.0 = completely equivalent
- 0.8 = very similar
- 0.5 = somewhat related
- 0.2 = weakly related
- 0.0 = unrelated or contradictory

---

**Prompt A Triplets:**
{triplets_a}

**Prompt B Triplets:**
{triplets_b}

---

## Please Return:

1. A list of aligned triplet pairs used in your comparison.
2. A JSON object with four fields:
   - `functional_equivalence`
   - `domain_type_equivalence`
   - `semantic_equivalence`
   - `entity_granularity_tolerance`
3. A short 1–2 sentence rationale for each score.

Format your response like: (make sure to use "[]" instead of "()" to have valid JSON)

```json
{{
  "aligned_triplets": [
    [["search", "flight", "Europe"], ["search", "flight", "international flights to Europe"]],
    [["target", "price_range", "cheapest"], ["target", "price_range", "low cost"]]
  ],
  "scores": {{
    "functional_equivalence": 1.0,
    "domain_type_equivalence": 0.8,
    "semantic_equivalence": 0.6,
    "entity_granularity_tolerance": 0.7
  }},
  "rationale": {{
    "functional_equivalence": "...",
    "domain_type_equivalence": "...",
    "semantic_equivalence": "...",
    "entity_granularity_tolerance": "..."
  }}
}}
```
'''

def obels_score(triplets_a, triplets_b, llm_model, args):
    prompt = OBELS_PROMPT_TEMPLATE.format(triplets_a=triplets_a, triplets_b=triplets_b)
    reply = get_llm_response(llm_model, prompt, args)
    match = re.search(r"```json\s*(.*?)\s*```", reply, re.DOTALL)
    if match:
        json_text = match.group(1)
    else:
        json_text = reply
    try:
        scores = json.loads(json_text)
        return scores["scores"]
    except Exception as e:
        print(f"OBELS parse error: {e}")
        return {"functional_equivalence": None, "domain_type_equivalence": None, "semantic_equivalence": None, "entity_granularity_tolerance": None}
    finally:
        time.sleep(args.llm_sleep)

# Add triplet extraction prompt and function
TRIPLET_EXTRACTION_PROMPT = '''You are an AI system that abstracts natural language prompts into semantic triplets of the form:
(intent, source_type, entity)

Where:
- `intent` captures the user's primary semantic goal. Instead of using a generic label like "get_info", choose more specific intents when appropriate. Use one of:
  - "learn": to understand or gain knowledge about a topic
  - "explore": to investigate options or alternatives
  - "analyze": to understand causes, effects, or implications
  - "compare": to contrast two or more entities or options
  - "summarize": to find concise descriptions or overviews
  - "plan": to organize steps toward a future action
  - "decide": to weigh alternatives with the goal of making a choice
  - "book", "watch", "read", "evaluate", etc. as appropriate
  - Retain "get_info" only for truly generic factual lookups

- `source_type` is the type of domain, service, or information requested (e.g., travel, symptom, policy_area, cooking_method, event, visa_process, treatment_method, academic_field, cuisine, etc).

- `entity` is the specific concept, item, or group of interest (e.g., Italy, depression, cold turkey, face transplant, immigration, Dulles airport, cold turkey, PhD in Business, Swahili dish, etc).

- If the prompt includes modifiers such as price, audience, date, location, or purpose, extract separate `target` triplets:
  ("target", source_type, entity)

Your output should be a list of triplets capturing each atomic semantic intent.
'''

def extract_triplets(prompt, llm_model, args):
    full_prompt = TRIPLET_EXTRACTION_PROMPT + f"\n**Prompt:**\n{prompt}\n**Triplets:**" 
    response = get_llm_response(llm_model, full_prompt, args)
    time.sleep(args.llm_sleep)
    return response

def Handle_OBELS_Scores(df, llm_model, args):
    print("Extracting triplets and computing OBELS metric...")
    obels_functional = []
    obels_domain_type = []
    obels_semantic = []
    obels_granularity = []
    orig_triplets_list = []
    recon_triplets_list = []
    for idx, row in df.iterrows():
        # Extract triplets
        orig_triplets = extract_triplets(row['prompts'], llm_model, args)
        orig_triplets_list.append(orig_triplets)
        recon_triplets = extract_triplets(row['reconstructed_prompt'], llm_model, args)
        recon_triplets_list.append(recon_triplets)
        # Compute OBELS metric
        obels = obels_score(orig_triplets, recon_triplets, llm_model, args)
        # Append results
        obels_functional.append(obels.get('functional_equivalence'))
        obels_domain_type.append(obels.get('domain_type_equivalence'))
        obels_semantic.append(obels.get('semantic_equivalence'))
        obels_granularity.append(obels.get('entity_granularity_tolerance'))
        print(f"OBELS [{idx+1}/{len(df)}]: F={obels.get('functional_equivalence')}, D={obels.get('domain_type_equivalence')}, S={obels.get('semantic_equivalence')}, G={obels.get('entity_granularity_tolerance')}")
    df['original_prompt_triplets'] = orig_triplets_list
    df['reconstructed_prompt_triplets'] = recon_triplets_list
    df['obels_functional_equivalence'] = obels_functional
    df['obels_domain_type_equivalence'] = obels_domain_type
    df['obels_semantic_equivalence'] = obels_semantic
    df['obels_entity_granularity_tolerance'] = obels_granularity

def print_and_save_results(df, args, input_file_name="", input_folder_path=""):
   
    avg_sbert = df['sbert_similarity'].mean()
    avg_llm = df['llm_judge_similarity'].mean()
    avg_obels_functional = df['obels_functional_equivalence'].mean()
    avg_obels_domain_type = df['obels_domain_type_equivalence'].mean()
    avg_obels_semantic = df['obels_semantic_equivalence'].mean()
    avg_obels_granularity = df['obels_entity_granularity_tolerance'].mean()
    
    print(f"Results for {input_file_name}:")
    print(f"Average SBERT similarity: {avg_sbert:.4f}")
    print(f"Average LLM judge similarity: {avg_llm:.4f}")
    print(f"Average OBELS functional equivalence: {avg_obels_functional:.4f}")
    print(f"Average OBELS domain type equivalence: {avg_obels_domain_type:.4f}")
    print(f"Average OBELS semantic equivalence: {avg_obels_semantic:.4f}")
    print(f"Average OBELS entity granularity tolerance: {avg_obels_granularity:.4f}")

    # Save results DataFrame to CSV
    output_filename = f"results_{os.path.basename(input_file_name)}"
    output_sub_folder = os.path.basename(input_folder_path)
    if not os.path.exists(os.path.join(args.output_dir, output_sub_folder)):
        os.makedirs(os.path.join(args.output_dir, output_sub_folder))
    output_path = os.path.join(args.output_dir, output_sub_folder, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Append averages to the summary CSV file
    summary_file = os.path.join(args.output_dir, output_sub_folder, "summary_results.csv")
    summary_data = {
        'file': input_file_name,
        'avg_sbert_similarity': avg_sbert,
        'avg_llm_judge_similarity': avg_llm,
        'avg_obels_functional_equivalence': avg_obels_functional,
        'avg_obels_domain_type_equivalence': avg_obels_domain_type,
        'avg_obels_semantic_equivalence': avg_obels_semantic,
        'avg_obels_entity_granularity_tolerance': avg_obels_granularity
    }
    
    summary_df = pd.DataFrame([summary_data])
    if os.path.exists(summary_file):
        summary_df.to_csv(summary_file, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(summary_file, index=False)
    
    print(f"Summary appended to: {summary_file}")
    
    # Save folder-level results to JSON if input folder is provided
    
    folder_results_file = os.path.join(args.output_dir, output_sub_folder, "folder_averages.json")
    
    # Load existing results if file exists
    folder_results = {}
    if os.path.exists(folder_results_file):
        try:
            with open(folder_results_file, 'r') as f:
                folder_results = json.load(f)
        except json.JSONDecodeError:
            folder_results = {}
    
    # Add current file results
    folder_results[input_file_name] = {
        'avg_sbert_similarity': float(avg_sbert),
        'avg_llm_judge_similarity': float(avg_llm),
        'avg_obels_functional_equivalence': float(avg_obels_functional),
        'avg_obels_domain_type_equivalence': float(avg_obels_domain_type),
        'avg_obels_semantic_equivalence': float(avg_obels_semantic),
        'avg_obels_entity_granularity_tolerance': float(avg_obels_granularity),
        'num_samples': len(df)
    }
        
    # Save to JSON file
    with open(folder_results_file, 'w') as f:
        json.dump(folder_results, f, indent=2)
    
    print(f"Folder results saved to: {folder_results_file}")

    print("Done.")
    
    # Return the results for potential use by HandleWholeFolder
    return {
        'file': input_file_name,
        'avg_sbert_similarity': avg_sbert,
        'avg_llm_judge_similarity': avg_llm,
        'avg_obels_functional_equivalence': avg_obels_functional,
        'avg_obels_domain_type_equivalence': avg_obels_domain_type,
        'avg_obels_semantic_equivalence': avg_obels_semantic,
        'avg_obels_entity_granularity_tolerance': avg_obels_granularity,
        'num_samples': len(df)
    }
    



def process_single_file(file_path, llm_model, sbert_model, args):
    """Process a single CSV file and return the results DataFrame"""
    print(f"\nProcessing file: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Handle unnamed columns - check for specific unnamed column patterns
    if 'prompts' not in df.columns:
        # Look for unnamed columns (pandas assigns 'Unnamed: 0', 'Unnamed: 1', etc.)
        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
        #unnamed_cols = [col for col in df.columns if col.startswith('original')]
        if unnamed_cols:
            # Use the first unnamed column as prompts
            first_unnamed = unnamed_cols[0]
            df = df.rename(columns={first_unnamed: 'prompts'})
            print(f"Renamed unnamed column '{first_unnamed}' to 'prompts'")
    
    if 'reconstructed_prompt' not in df.columns:
        # Look for unnamed columns (pandas assigns 'Unnamed: 0', 'Unnamed: 1', etc.)
        unnamed_cols = [col for col in df.columns if col.startswith('reconstruction')]
        if unnamed_cols:
            # Use the first unnamed column as prompts
            first_unnamed = unnamed_cols[0]
            df = df.rename(columns={first_unnamed: 'reconstructed_prompt'})
            print(f"Renamed unnamed column '{first_unnamed}' to 'reconstructed_prompt'")

    print("Computing SBERT similarity...")
    df['sbert_similarity'] = compute_sbert_similarity(df, sbert_model)

    print("Computing LLM judge similarity...")
    df['llm_judge_similarity'] = Compute_LLM_Judge_Similarity(df, llm_model, args)

    Handle_OBELS_Scores(df, llm_model, args)
    
    return df


def HandleWholeFolder(input_folder, llm_model, sbert_model, args):
    """Process all CSV files in a folder and calculate folder-level averages"""
    print(f"Processing folder: {input_folder}")
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    all_results = []
    
    for file_path in csv_files:
        #try:
            # Check if file has required columns before processing
            df = pd.read_csv(file_path)
            
            # Check for prompts column or unnamed column
            has_prompts = 'prompts' in df.columns
            has_reconstructed = 'reconstructed_prompt' in df.columns
            
            # Check for unnamed columns (Unnamed: 0, Unnamed: 1, etc.)
            unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:') or col.startswith('reconstruction')]
            has_unnamed = len(unnamed_cols) > 0
            
            missing_columns = []
            if not has_prompts and not has_unnamed:
                missing_columns.append('prompts')
            if not has_reconstructed and not has_unnamed:
                missing_columns.append('reconstructed_prompt')
            
            if missing_columns:
                print(f"Skipping {os.path.basename(file_path)}: Missing required columns: {missing_columns}")
                print(f"Available columns: {list(df.columns)}")
                continue
            
            df = process_single_file(file_path, llm_model, sbert_model, args)
            result = print_and_save_results(df, args, os.path.basename(file_path), input_folder)
            
            # Ensure result has required keys
            if result and 'avg_sbert_similarity' in result and 'num_samples' in result:
                all_results.append(result)
            else:
                print(f"Warning: Invalid result structure for {os.path.basename(file_path)}")
                
        # except Exception as e:
        #     print(f"Error processing {file_path}: {e}")
        #     continue
    
    # Calculate and save overall folder averages
    if all_results:
        output_sub_folder = os.path.basename(input_folder)
        folder_results_file = os.path.join(args.output_dir, output_sub_folder, "folder_averages.json")
        
        # Load existing results
        folder_results = {}
        if os.path.exists(folder_results_file):
            try:
                with open(folder_results_file, 'r') as f:
                    folder_results = json.load(f)
            except json.JSONDecodeError:
                folder_results = {}
        
        # Calculate overall folder averages
        total_sbert = sum(r['avg_sbert_similarity'] for r in all_results)
        total_llm = sum(r['avg_llm_judge_similarity'] for r in all_results)
        total_obels_functional = sum(r['avg_obels_functional_equivalence'] for r in all_results)
        total_obels_domain_type = sum(r['avg_obels_domain_type_equivalence'] for r in all_results)
        total_obels_semantic = sum(r['avg_obels_semantic_equivalence'] for r in all_results)
        total_obels_granularity = sum(r['avg_obels_entity_granularity_tolerance'] for r in all_results)
        total_samples = sum(r['num_samples'] for r in all_results)
        num_files = len(all_results)
        
        overall_folder_avg = {
            'avg_sbert_similarity': total_sbert / num_files,
            'avg_llm_judge_similarity': total_llm / num_files,
            'avg_obels_functional_equivalence': total_obels_functional / num_files,
            'avg_obels_domain_type_equivalence': total_obels_domain_type / num_files,
            'avg_obels_semantic_equivalence': total_obels_semantic / num_files,
            'avg_obels_entity_granularity_tolerance': total_obels_granularity / num_files,
            'total_samples': total_samples,
            'num_files': num_files,
            'processed_files': [r['file'] for r in all_results]
        }
        
        # Add overall averages to the JSON
        folder_results['overall_folder_averages'] = overall_folder_avg
        
        # Save updated JSON
        with open(folder_results_file, 'w') as f:
            json.dump(folder_results, f, indent=2)
        
        print(f"\nOverall folder results:")
        print(f"Average SBERT similarity across all files: {overall_folder_avg['avg_sbert_similarity']:.4f}")
        print(f"Total samples processed: {overall_folder_avg['total_samples']}")
        print(f"Number of files processed: {overall_folder_avg['num_files']}")
        print(f"Updated folder results saved to: {folder_results_file}")
    else:
        print("No files were successfully processed.")