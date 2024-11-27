import argparse
from enum import Enum
import torch
from typing import List
import random
import json

from accelerate import init_empty_weights
import outlines
import outlines.caching as cache

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, pipeline
from tqdm.auto import tqdm
from pydantic import BaseModel, ValidationError, conlist,constr

from config import hf_token
from evaluate import calculate_statistics,generate_validation_regex
from utils import serialize_dict

# Disabling outlines cache because it balloons to 300+ GB
cache.disable_cache()

prompt_dict = {"reasoning" : """You are an expert problem solver. You are doing the "connections" puzzle. You will receive 16 words.
    Find groups of four items that share something in common.

    Select four items and tap 'Submit' to check if your guess is correct.

    Here is an example with 2 groups of 4 fours words

    Your words are: Bass;Opal;Trout;Salmon,Ant,Drill,Island;Flounder

    Create a JSON with your reasoning and the groups of four found.

    {'reasoning' : 'Bass, Flounder, Salmon, Trout are all fish. For ther other words the pattern is harder to see, but it looks like each can have the word fire in front' ,
    'groups' : [{'theme' : 'FISH', 'words' : ['Bass', 'Flounder', 'Salmon', 'Trout']},
    {'theme' : 'FIRE __', 'words' : ['Ant', 'Drill', 'Island', 'Opal']}]}

    Categories will always be more specific than "5-LETTER-WORDS," "NAMES" or "VERBS."

    Each puzzle has exactly one solution. Watch out for words that seem to belong to multiple categories!
    """,
    "default":
    """You are an expert problem solver. You are doing the "connections" puzzle. You will receive 16 words.
    Find groups of four items that share something in common.

    Select four items and tap 'Submit' to check if your guess is correct.

    Here is an example with 2 groups of 4 fours words

    Your words are: Bass;Opal;Trout;Salmon,Ant,Drill,Island;Flounder

    Create a JSON with your reasoning and the groups of four found.

    {'groups' : [{'theme' : 'FISH', 'words' : ['Bass', 'Flounder', 'Salmon', 'Trout']},
    {'theme' : 'FIRE __', 'words' : ['Ant', 'Drill', 'Island', 'Opal']}]}

    Categories will always be more specific than "5-LETTER-WORDS," "NAMES" or "VERBS."

    Each puzzle has exactly one solution. Watch out for words that seem to belong to multiple categories!
    """
    }


    
def parse_args():
    parser = argparse.ArgumentParser(description="Parse constants for model and prompt settings.")

    # Seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Model ID with choices
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        choices = [
                            "mistralai/Mistral-7B-Instruct-v0.3",
                            "meta-llama/Llama-3.2-1B-Instruct",
                            "meta-llama/Llama-3.2-3B-Instruct",
                            "meta-llama/Meta-Llama-3.1-8B-Instruct",
                            "Qwen/Qwen2.5-1.5B-Instruct",
                            "Qwen/Qwen2.5-3B-Instruct",
                            "Qwen/Qwen2.5-14B-Instruct",
                        ],
                        help='Huggingface Model Path')

    # Prompt settings
    parser.add_argument('--use_structured_prediction', action='store_true', 
                        default=False, help='Enable structured prediction')
    parser.add_argument('--prompt_version', type=str, default="default", 
                        choices=["reasoning", "default"], 
                        help='Prompt version to use')
    parser.add_argument('--k_shot', type=int, default=0, 
                        help='Number of shots for few-shot learning. Must be less than sample_size - 1')

    # Data settings
    parser.add_argument('--sample_size', type=int, default=-1, 
                        help='Sample size. Valid choices are -1 or greater')

    # Generation settings
    parser.add_argument('--temperature', type=float, default=0.0, 
                        help='Sampling temperature. Range: 0 to 1')
    parser.add_argument('--top_p', type=float, default=1.0, 
                        help='Top-p sampling probability. Range: 0 to 1')
    parser.add_argument('--resolution', type=int, default =16, choices= [16,8,4],
                        help='Bits to quantize model with')
    parser.add_argument('--max_tokens', type=int, default =1024,
                        help='')

    return parser.parse_args()
    
    
if __name__ == "__main__":
    from datetime import date
    import time
    
    args = parse_args()
    seed = args.seed
    model_id = args.model_id
    use_structured_prediction = args.use_structured_prediction
    prompt_version = args.prompt_version
    k_shot = args.k_shot
    sample_size = args.sample_size
    temperature = args.temperature
    top_p = args.top_p
    resolution = args.resolution
    max_tokens = args.max_tokens
    # Init random seeds
    set_seed(seed)
    
    # Init model
    # TODO: Make it so that the model can load into both CPU and GPU Ram by default 
    # to avoid OOMs when running models that are smaller than GPU, but large enough
    # to cause issues at decoding time
    if resolution == 16:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                                   trust_remote_code=True,
                                                                   attn_implementation="flash_attention_2",
                                                                   token=hf_token,
                                                                   device_map="auto")
    elif resolution == 8:
        model = AutoModelForCausalLM.from_pretrained(model_id,load_in_8bit=True,
                                                                   trust_remote_code=True,
                                                                   attn_implementation="flash_attention_2",
                                                                   token=hf_token,
                                                                   device_map="auto")            
    elif resolution == 4:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,
                                                                   trust_remote_code=True,
                                                                   attn_implementation="flash_attention_2",
                                                                   token=hf_token,
                                                                   device_map="auto")    
    else:
        raise NotImplementedError()
        
    tokenizer = AutoTokenizer.from_pretrained(model_id,token=hf_token)
    
    if not use_structured_prediction:
        instruction_pipeline = pipeline("text-generation", model=model,
        tokenizer=tokenizer, max_new_tokens=max_tokens)
    else:
        outline_model = outlines.models.Transformers(model=model, tokenizer=tokenizer)
        if temperature == 0.0:
            sampler = outlines.samplers.greedy()
        elif temperature > 0 and temperature <= 1.0:
            sampler = outlines.samplers.multinomial(temperature=temperature,top_p=top_p)
        else:
            raise ValueError(f"Unsupported temperature {temperature} received, expected a value between 0 and 1")
            
    # Set prompt
    connections_prompt = prompt_dict[prompt_version]

    results = {}
    nyt_connections_data = json.load(open("NYT-Connections-Answers/connections.json",'r'))
    if sample_size > 0:
        nyt_connections_data = [val for val in random.Random(seed).sample(nyt_connections_data,sample_size)]
    else:
        pass
    start_time = time.perf_counter()
    for datum_idx, nyt_connections_datum in enumerate(tqdm(nyt_connections_data)):
        

        messages = [{"role" : "system", "content" : connections_prompt}]
        if k_shot > 0:
            # TODO: Implement k shot
            examples = nyt_connections_data[:datum_idx] + nyt_connections_data[datum_idx+1:]
            k_shot_examples = random.Random(seed).sample(examples,k_shot)
            for k_shot_example in k_shot_examples:
                k_shot_connections_words = [word for item in k_shot_example['answers'] for word in item['members']]
                formatted_example = {"groups" : [{"words" : group['members'],"theme" : group['group']} for group in k_shot_example["answers"]]}
                messages.append({"role" : "user", "content" : f"Your words are {";".join(k_shot_connections_words)}. Good luck!"})
                messages.append({"role" : "assistant", "content" : json.dumps(formatted_example)})
            
        connections_words = [word for item in nyt_connections_datum['answers'] for word in item['members']]
        random.Random(seed).shuffle(connections_words)
        prompt_words = f"Your words are {";".join(connections_words)}. Good luck!"
        messages += [{"role" : "user", "content" : prompt_words}]

        if use_structured_prediction:
            AllowedWords = Enum('AllowedWords', {val: val for val in connections_words},type=str)
            
            class Group(BaseModel):
                words: conlist(AllowedWords,min_length=4,max_length=4)
                theme: str
            
            if prompt_version == "reasoning":
                class ConnectionSolution(BaseModel):
                    reasoning : constr(max_length=256)
                    groups : conlist(Group,min_length=4,max_length=4)
            else:
                class ConnectionSolution(BaseModel):
                    groups : conlist(Group,min_length=4,max_length=4)

            generator = outlines.generate.json(outline_model, schema_object=ConnectionSolution,sampler=sampler)
            try:
                connection_results = generator("\n".join([message['content'] for message in messages]), seed=seed,max_tokens=max_tokens)
                results[nyt_connections_datum['id']] = json.loads(connection_results.json())
            except Exception as e:
                print(e)
        else:
            if temperature == 0.0:
                generated_output = instruction_pipeline(messages, temperature=temperature, do_sample=False,seed=seed)
                generated_text = generated_output[0]['generated_text'][-1]['content']
                try:
                    results[nyt_connections_datum['id']] = json.loads(generated_text)
                except:
                    results[nyt_connections_datum['id']] =  generated_text
            elif temperature > 0 and temperature <= 1.0:
                generated_output = instruction_pipeline(messages, temperature=temperature, top_p=top_p,seed=seed)
                generated_text = generated_output[0]['generated_text'][-1]['content']
                try:
                    results[nyt_connections_datum['id']] = json.loads(generated_text)
                except:
                    results[nyt_connections_datum['id']] =  generated_text
            else:
                raise ValueError(f"Unsupported temperature {temperature} received, expected a value between 0 and 1")
    end_time = time.perf_counter()
    args_dict = vars(args)
    serialized_args = serialize_dict(args_dict)
    output = {"results" :results,
              "metadata" : {
              "total_run_time_seconds" : end_time - start_time,
              "write_date" : date.today().strftime("%Y-%m-%d"),
              "parameters" : args_dict,
              "code_version" : "0.0.2"},
              "metrics" : calculate_statistics(results),
              }
    json.dump(output,open(f'results/experiment-{serialized_args}.json','w'))