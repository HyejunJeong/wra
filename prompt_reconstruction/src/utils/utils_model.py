# import unsloth
# from unsloth import FastLanguageModel
# from unsloth.chat_templates import get_chat_template
from openai import OpenAI
import google.generativeai as genai
from google.generativeai import types
from anthropic import Anthropic
import torch

def get_llm_model(args):
    if args.model == "gpt-4o" or args.model == "gpt-4.1-nano" or args.model == "gpt-5" or args.model == "gpt-5-mini" or args.model == "gpt-4o-mini":   
        # Create OpenAI client with API key
        client = OpenAI(api_key=args.openai_api_key)
        # Store model and temperature for later use
        client.model = args.model
        client.temperature = args.temperature
        
        return client
    elif args.model == "gpt-4o-finetuned":
        client = OpenAI(api_key=args.openai_api_key)
        client.model = "ft:gpt-4o-2024-08-06:umass-amherst-s-p-projects:prompt-recon-full-training-set:C1hbuZGC"
        client.temperature = args.temperature
        return client
    elif args.model == "gpt-4.1-nano-finetuned":
        client = OpenAI(api_key=args.openai_api_key)
        client.model = "ft:gpt-4.1-nano-2025-04-14:umass-amherst-s-p-projects:prompt-recon-full-training-set:C1hOc9aJ"
        client.temperature = args.temperature
        return client
    elif args.model == "gemini-2.5-pro":
        # Configure the Gemini API
        genai.configure(api_key=args.gemini_api_key)
        # Create a client object with model and temperature
        client = type('obj', (object,), {
            'model': genai.GenerativeModel(args.model),  # Use the correct model identifier
            'temperature': args.temperature
        })
        return client
    elif args.model == "claude-opus-4-1-20250805":
        client = Anthropic(api_key=args.claude_api_key)
        client.model = args.model
        client.temperature = args.temperature
        return client
    elif args.model == "Llama-3.1-8B-Instruct":
        
        # Load Llama model using unsloth for optimized inference
        max_seq_length = 5000
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/datasets/ai/llama3/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        # Store model, tokenizer, and temperature for later use
        client = type('obj', (object,), {
            'model': model,
            'tokenizer': tokenizer,
            'temperature': args.temperature
        })
        return client
    elif args.model == "Gemma-3-4b":
        # Load Gemma model using unsloth for optimized inference
        max_seq_length = 5000
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/home/mteymoorianf_umass_edu/.cache/huggingface/hub/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe",
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        # Apply Gemma-3 chat template
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        # Store model, tokenizer, and temperature for later use
        client = type('obj', (object,), {
            'model': model,
            'tokenizer': tokenizer,
            'temperature': args.temperature
        })
        return client
    elif args.model == "Qwen-2.5-32B-Instruct":
        # Load Qwen model using unsloth for optimized inference
        max_seq_length = 5000
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "/datasets/ai/qwen/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True
        )
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")
        # Store model, tokenizer, and temperature for later use
        client = type('obj', (object,), {
            'model': model,
            'tokenizer': tokenizer,
            'temperature': args.temperature
        })
        return client
    else:
        raise ValueError(f"Invalid LLM model: {args.model}. Supported models: 'gpt-4o', 'gpt-4.1-nano', 'Llama-3.1-8B-Instruct', 'gemini-2.5-pro', 'claude-opus-4-1-20250805', 'Gemma-3-2B', 'Qwen-2.5-32B-Instruct'")

def Llama_3_1_8B_Instruct(llm_model, prompt, args):
    # Generate response using Llama model
    inputs_ids = llm_model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(llm_model.model.device)
    
    with torch.no_grad():
        outputs = llm_model.model.generate(
            input_ids=inputs_ids,
            max_new_tokens=15000,
            do_sample=False,
            temperature=llm_model.temperature,
            top_p=0.9
        )
    
    output = llm_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the reconstructed prompt from the assistant's response
    # The model generates a full conversation, we need just the assistant's answer
    
    # Method 2: Look for the assistant section
    if "Reconstructed Prompt:assistant" in output:
        assistant_start = output.find("Reconstructed Prompt:assistant")
        if assistant_start != -1:
            # Get everything after "assistant" and clean it up
            assistant_response = output[assistant_start:].split("assistant", 1)[1].strip()
            # Remove any remaining system/user messages if they appear
            if "user:" in assistant_response:
                assistant_response = assistant_response.split("user:")[0].strip()
            if "system:" in assistant_response:
                assistant_response = assistant_response.split("system:")[0].strip()
            return assistant_response
    
    # Method 3: Fallback - return everything after the input prompt
    response = output[len(prompt):].strip()
    return response

def Gemma_3_4B(llm_model, prompt, args):
    # Generate response using Gemma model
    text = llm_model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,  # Must add for generation
    )
    with torch.no_grad():
        outputs = llm_model.model.generate(
            **llm_model.tokenizer([text], return_tensors = "pt").to(llm_model.model.device),
            max_new_tokens = 15000, # Increase for longer outputs!
            # Recommended Gemma-3 settings!
            temperature = llm_model.temperature, top_p = 0.95, top_k = 64,
        )
    output = llm_model.tokenizer.batch_decode(outputs)[0]
    
    if "<start_of_turn>model" in output:
        model_start = output.find("<start_of_turn>model")
        if model_start != -1:
            # Get everything after "<start_of_turn>model" and clean it up
            model_response = output[model_start:].split("<start_of_turn>model", 1)[1].strip()
            # Remove the end marker if present
            if "<end_of_turn>" in model_response:
                model_response = model_response.split("<end_of_turn>")[0].strip()
            return model_response
    
def Qwen_2_5_32B_Instruct(llm_model, prompt, args):
    # Format input with chat template → returns plain string
    text = llm_model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = llm_model.tokenizer([text], return_tensors="pt").to(llm_model.model.device)

    with torch.no_grad():
        outputs = llm_model.model.generate(
            **inputs,
            max_new_tokens=15000,   # safer default than 5000
            temperature=llm_model.temperature,
            top_p=0.9,
            do_sample=True
        )

    # Decode
    output = llm_model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    response = output.split("Reconstructed Prompt:")[-1].strip()
    # remove leading 'assistant' if present
    if response.startswith("assistant"):
        response = response.replace("assistant", "", 1).strip()
    return response

    

def get_llm_response(llm_model, prompt, args):
    if args.model == "gpt-4o" or args.model == "gpt-4.1-nano" or args.model == "gpt-5" or args.model == "gpt-5-mini" or args.model == "gpt-4o-mini" or args.model == "gpt-4o-finetuned" or args.model == "gpt-4.1-nano-finetuned":   
        # GPT-5 models only support default temperature (1), so we need to handle this
        if args.model == "gpt-5" or args.model == "gpt-5-mini":
            response = llm_model.chat.completions.create(
                model=llm_model.model,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            response = llm_model.chat.completions.create(
                model=llm_model.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_model.temperature,
                max_completion_tokens=args.max_tokens
            )
        return response.choices[0].message.content
    elif args.model == "gemini-2.5-pro":
        # Use the Gemini model directly
        response = llm_model.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=llm_model.temperature,
            )
        )
        return response.text
    elif args.model == "claude-opus-4-1-20250805":
        response = llm_model.messages.create(
            model=args.model,  # or an alias like "claude-3-7-sonnet-latest"
            max_tokens=args.max_tokens,                    # REQUIRED
            temperature=llm_model.temperature,                   # optional
            messages=[{"role": "user", "content": prompt}],
        )
        response = "".join(b.text for b in response.content if b.type == "text")
        return response
    elif args.model == "Llama-3.1-8B-Instruct":
        return Llama_3_1_8B_Instruct(llm_model, prompt, args)
    elif args.model == "Gemma-3-4b":
        return Gemma_3_4B(llm_model, prompt, args)
    elif args.model == "Qwen-2.5-32B-Instruct":
        return Qwen_2_5_32B_Instruct(llm_model, prompt, args)
    else:
        raise ValueError(f"Invalid LLM model: {args.model}. Supported models: 'gpt-4o', 'gpt-5', 'gpt-5-mini', 'gpt-4o-mini', 'gemini-2.5-pro', 'claude-opus-4-1-20250805', 'Llama-3.1-8B-Instruct', 'Gemma-3-27B', 'Qwen-2.5-32B-Instruct'")


