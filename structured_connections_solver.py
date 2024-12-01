from collections import Counter, defaultdict
from itertools import combinations, permutations, product
import json
import random
from statistics import mean

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed,pipeline
from tqdm.auto import tqdm

from config import hf_token
from utils import serialize_dict

# Constants
default_system_prompt = "Given two words, give a yes or no answer as to whether or not the words have a relationship."
theme_aware_system_prompt = """Given two words, give a yes or no answer as to whether or not the words have a relationship.
The types of relationships can include, but are not limited to
1. Synonyms
2. Homophones
3. Sharing a leading or train word
4. Some common usage
5. Names of things in a similar group
6. Physical similarities
7. Anagrams

You are an expert linguistic, so please be as confident as possible. If there are no obvious connections, say no"""

system_prompt_dict = {
"default" : default_system_prompt,
"themed": theme_aware_system_prompt,
}

pair_prompt = "The words are: {};{}. Are they related?"
quad_prompt = "The words are: {};{};{};{}. Are they related?"

aggregation_dict = {"min" : min,"max" : max,"mean": mean, "first" : lambda x: x[0], "last" : lambda x:[-1]}

def format_messages(messages):
    formatted_messages = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
    return formatted_messages

def batcher(lst, batch_size):
    for i in range(0,len(lst),batch_size):
        yield lst[i:i+batch_size]
        
negative_pairs = [pair for group1_idx,group2_idx in combinations(range(4),2) for pair in product(*[range(group1_idx*4,group1_idx*4+4),range(group2_idx*4,group2_idx*4+4)])]
# items follows the format above
def create_k_shot_examples(items, question_prompt):
    k_shot_examples = []
    for item in items:
        yes_answer_idx, yes_idx1,yes_idx2 = random.randint(0,3), random.randint(0,3),random.randint(0,3)
        yes_members = item['answers'][yes_answer_idx]['members']
        no_idx1, no_idx2 = random.sample(negative_pairs,1)[0]
        negative_pair = item['answers'][no_idx1//4]['members'][no_idx1%4],item['answers'][no_idx2//4]['members'][no_idx2%4]
        positive_pair = yes_members[yes_idx1],yes_members[yes_idx2]
        k_shot_examples.append({"role": "user", "content" :question_prompt.format(*negative_pair)})
        k_shot_examples.append({"role": "assistant", "content" :"No"})
        k_shot_examples.append({"role": "user", "content" :question_prompt.format(*positive_pair)})
        k_shot_examples.append({"role": "assistant", "content" :"Yes"})     
    return k_shot_examples

def create_system_message(words, system_prompt, question_prompt,k_shot_examples):
    messages = [{"role" : "system", "content" : system_prompt}]
    messages += k_shot_examples
    messages.append({"role": "user", "content" :question_prompt.format(*words)})
    input_text = format_messages(messages)
    return input_text

def generate_keys(iterable, generation_chunk_size):
    for pair in combinations(iterable,generation_chunk_size):
        yield tuple(sorted(pair))
        
def evaluate(words,sims,generation_chunk_size):
    return np.prod([sims[idx_combination] for group in batcher(words,4) for idx_combination in generate_keys(group,generation_chunk_size)])

# TODO: REWORK SIMILARITY
def local_search(sims, generation_chunk_size, k =10, patience=0, search_type="greedy",top_p = 1.0, search_random_top_k=-1):
    idx_tuples_to_search = [tuple(random.sample(words, 16)) for _ in range(k)]
    idx_scores = [evaluate(idx_tuple_to_search,sims,generation_chunk_size) for idx_tuple_to_search in idx_tuples_to_search]
    max_score = max(idx_scores)
    idx_score_mapping = {idx_tuple_to_search: idx_score for idx_tuple_to_search, idx_score in zip(idx_tuples_to_search,idx_scores)}
    searching = True
    turns_without_improvement = 0
    def swap_pos(tpl,idx1,idx2):
        lst = list(tpl)
        tmp = lst[idx1]
        lst[idx1] = lst[idx2]
        lst[idx2] = tmp
        return tuple(lst)
        
    while searching:
        # TODO: adapt this for 4 position
        idx_permutation_tuples = [swap_pos(idx_tuple_to_search,idx1,idx2) for (idx1,idx2) in generate_keys(range(16),generation_chunk_size) for idx_tuple_to_search in idx_tuples_to_search] # TODO: Create iteration tuples from idxs_to_search, remove dupes
        for idx_permutation_tuple in idx_permutation_tuples:
            if idx_permutation_tuple in idx_score_mapping:
                pass
            else:
                idx_score_mapping[idx_permutation_tuple] = evaluate(idx_permutation_tuple,sims,generation_chunk_size)
        
        idx_score_mapping = dict(sorted(idx_score_mapping.items(), key=lambda item: item[1],reverse=True))
        # TODO: Look at other selection methods, currently greedy by default
        if search_type == "greedy":
            new_idx_tuples_to_search, idx_scores = zip(*[item for item, _ in zip(idx_score_mapping.items(),range(k))])
        elif search_type == "random":
            #import pdb;pdb.set_trace()
            filtered_items = idx_score_mapping.items()
            if top_p < 1.0:
                raise NotImplementedError()
            if search_random_top_k > 0:
                filtered_items = [tpl for tpl,_ in zip(idx_score_mapping.items(),range(search_random_top_k))]
            new_idx_tuples_to_search, idx_scores = zip(*random.sample(filtered_items,k))
        else:
            raise NotImplementedError(f"Search type {search_type} not implemented")
            
        max_new_score = max(idx_scores)
        if set(idx_tuples_to_search) == set(new_idx_tuples_to_search): # CHECK IF LISTS ARE OVERLAPPED, MAYBE USE SET
            searching=False
        elif max_new_score > max_score:
            turns_without_improvement = 0
            max_score = max_new_score
            idx_tuples_to_search = new_idx_tuples_to_search
        else:
            turns_without_improvement += 1
            if turns_without_improvement > patience:
                searching = False
        
    # TODO: Extract max score from idx_score_mapping (we could search other)
    return idx_tuples_to_search[0],idx_scores[0]

def capped_iterator(iterator, cap):
    counter = defaultdict(int)
    for val in iterator:
        counter[frozenset(val)] += 1 
        if counter[frozenset(val)] <= cap:
            yield val

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Parse constants for model and prompt settings.")

    # Seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # ModelSettings
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        choices = [
                            "meta-llama/Llama-3.2-1B-Instruct",
                            "meta-llama/Llama-3.2-3B-Instruct",
                            "meta-llama/Meta-Llama-3.1-8B-Instruct",
                            "Qwen/Qwen2.5-1.5B-Instruct",
                            "Qwen/Qwen2.5-3B-Instruct",
                            "Qwen/Qwen2.5-7B-Instruct",
                            "Qwen/Qwen2.5-14B-Instruct",
                        ],
                        help='Huggingface Model Path')
                                                
    parser.add_argument('--resolution', type=int, default =16, choices= [16,8,4],
                        help='Bits to quantize model with')

    parser.add_argument('--prompt_version', type=str, default="default", 
                        choices=system_prompt_dict.keys(), 
                        help='Prompt version to use')
                        
    # Data settings
    parser.add_argument('--sample_size', type=int, default=-1, 
                        help='Sample size. Valid choices are -1 or greater')

    # Generation settings
    
    parser.add_argument('--prompt_k', type=int,default=0,
                        help="Number of examples to include in prompt")
    
    parser.add_argument('--generation_chunk_size', type=int, default=2, choices=[2,4],
                        help='Number of items to search at once')
    
    # Search settings
                        
    parser.add_argument('--search_k', type=int, default=10, 
                        help='Number of items to search at once')

    parser.add_argument('--search_random_top_k', type=int, default=-1, 
                        help='Number of items to search at once')
                        
    parser.add_argument('--search_patience', type=int, default=0, 
                        help='Number of turns to wait before quitting search')

    parser.add_argument('--search_type', type=str, default="greedy", choices=["greedy","random"],
                        help='Search method to be used on model probabilities')
                        
    parser.add_argument('--aggregation_type', type=str, default="max", choices=aggregation_dict.keys(),
                        help='Aggregation method for similarities output by the model.')

    parser.add_argument('--search_perm_cap', type=int, default=2, 
                        help='Amount of permutations to keep for a given')
                        
                        
    return parser.parse_args()
    
if __name__ == "__main__":
    from datetime import date
    import os
    import time
    
    args = parse_args()
    
    args_dict = vars(args)
    serialized_args = serialize_dict(args_dict)
    '''
    if os.path.exists(f'results/experiment-{serialized_args}.json'):
        print("Exiting experiment with params: ", args_dict)
        exit()
    '''
    # TODO: Add check to skip if already computed
    
    seed = args.seed
    model_id = args.model_id
    resolution = args.resolution
    
    # FOOD FOR THOUGHT,
    # ONLY GENERATE WHAT YOU'RE SEARCHING RATHER THAN ALL PROBS
    
    # CONSTANT: CHANGE IF YOU ARE GETTING OOD. I suggest making it a factor of the number of perms generated
    # this is given by min(search_perm_cap*(16 choose generation_chunk_size), perm(16,generation_chunk_size))
    inference_batch_size = 20
    sample_size = args.sample_size
    system_prompt_type = args.prompt_version
    device = "auto"
    aggregation_type = args.aggregation_type
    
    generation_chunk_size = args.generation_chunk_size
    search_k = args.search_k
    search_patience = args.search_patience
    search_type = args.search_type # random, weighted ?? genetic?? ?? hungarian matching approach ?? maybe there is a deterministic way of doing theis
    search_random_top_k = args.search_random_top_k
    search_perm_cap = args.search_perm_cap
    
    prompt_k = args.prompt_k
    
    set_seed(seed)

    nyt_connections_data = json.load(open("NYT-Connections-Answers/connections.json",'r'))

    if sample_size > 0:
        nyt_connections_data = [val for val in random.Random(seed).sample(nyt_connections_data,sample_size)]
    
    # Device map to avoid OOMs
    # https://huggingface.co/docs/accelerate/v0.25.0/en/concept_guides/big_model_inference
    if resolution == 16:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                                   trust_remote_code=True,
                                                                   attn_implementation="flash_attention_2",
                                                                   token=hf_token,
                                                                   device_map=device,
                                                                   max_memory={0: "12GiB", "cpu": "30GiB"})
    elif resolution == 8:
        model = AutoModelForCausalLM.from_pretrained(model_id,load_in_8bit=True,
                                                                   trust_remote_code=True,
                                                                   attn_implementation="flash_attention_2",
                                                                   token=hf_token,
                                                                   device_map=device,
                                                                   max_memory={0: "12GiB", "cpu": "30GiB"})            
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
                                                                   device_map="auto",
                                                                   max_memory={0: "12GiB", "cpu": "30GiB"})
    else:
        raise NotImplementedError()
        
    tokenizer = AutoTokenizer.from_pretrained(model_id,token=hf_token, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # Create dataset from prompts
    # try with different settings 
    # k-shot 0 -> 5
    # presence of theme
        
    yes_idxs = [idx for word,idx in tokenizer.vocab.items() if word.replace('Ġ','').lower().strip() == "yes"]
    no_idxs = [idx for word,idx in tokenizer.vocab.items() if word.replace('Ġ','').lower().strip() == "no"]
    yes_no_idxs = yes_idxs + no_idxs

    results = {}
    start_time = time.perf_counter()
    raw_similarities = {}
    # This might seem odd, but the Qwen models have a mismatch
    # between vocab and output size
    if 'Qwen' in model_id:
        # NOTE: Hacky way of getting real embedding size vs vocab size
        # Qwen pads their embeddings to get a power of 2 divisble number and it's incompatible with the way HF stores vocab..
        # I tried different ways of extracting the output size, but couldn't get it to work consistently across resolutions, not sure why
        mask_size = len(tokenizer)
        p_delta = min([dim-mask_size for p in model.parameters() for dim in p.shape],key=abs)
        mask_size += p_delta
    else:
        mask_size = len(tokenizer)
            
    for datum_idx, datum in enumerate(tqdm(nyt_connections_data)):
        words = [word for answer in datum['answers'] for word in answer['members']]
        gt = [answer for answer in datum['answers']]
        random.shuffle(words)
        
        # Capping the number of permutations, the resulting number is high for 4
        all_chunk_idxs = [permuted_idxs for permuted_idxs in capped_iterator(permutations(range(16),generation_chunk_size),search_perm_cap)]
        all_word_chunks = [tuple([words[idx] for idx in chunk]) for chunk in all_chunk_idxs]
        
        # TODO: Select prompt_k examples
        prompt_to_use = quad_prompt if generation_chunk_size == 4 else pair_prompt
        
        selected_datums = random.Random(seed).sample(nyt_connections_data[:datum_idx] + nyt_connections_data[datum_idx+1:], prompt_k)
        
        k_shot_examples = create_k_shot_examples(selected_datums, prompt_to_use)
        input_texts = [create_system_message(chunk, system_prompt_dict[system_prompt_type], prompt_to_use, k_shot_examples) for chunk in all_word_chunks]
        tokenizer.pad_token = tokenizer.eos_token
        #import pdb;pdb.set_trace()
        model_outputs = []
            
        for input_text_batch in batcher(input_texts, inference_batch_size):
            inputs = tokenizer(input_text_batch, padding=True,return_tensors="pt").to(model.device)
            # Generate a response
            
            # This mi
            mask = torch.zeros(mask_size)
            mask[yes_no_idxs] = 1
            
            output = model.generate(
                **inputs,
                max_new_tokens=1,  # Set based on expected response length
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                top_p=None,
                temperature=None,
            )
            model_outputs.append(output['scores'][0])
        
        output_probs = torch.softmax(torch.vstack(model_outputs),dim=1).to("cpu")
        premask_probs = output_probs[:,yes_no_idxs].sum(axis=1)
        masked_probs = output_probs * mask
        masked_probs /= masked_probs.sum(axis=1).reshape(-1,1).tile(1,masked_probs.shape[1])
        yes_prob = masked_probs[:,yes_idxs].sum(axis=1)
        
        # TODO: REPORT GT SCORES VS NON-GT SCORES
        # THIS COULD HELP ID ISSUES AND GUIDE PROMPTS
        scores = yes_prob.tolist()
        raw_similarity = defaultdict(list)
        for all_word_chunk,score in zip(all_word_chunks,scores):
            raw_similarity[tuple(sorted(all_word_chunk))].append(score)
        
        raw_similarities[str(datum['id'])] = [[w1,w2,sim] for (w1,w2),sim in raw_similarity.items()]
        
        sim_dict = {}
        aggregate = aggregation_dict[aggregation_type]
        for word_tuple,current_scores in raw_similarity.items():
            sim_dict[word_tuple] = aggregate(current_scores)

        best_grouping, grouping_score = local_search(sim_dict,generation_chunk_size, k=search_k,patience=search_patience,search_type=search_type,search_random_top_k=search_random_top_k)
        proposed_groupings = [group for group in batcher(best_grouping,4)]
        
        # TODO: Add sims to metadata, maybe have a verbose mode
        # todo: add time for search, time for decoding, etc..
        results[str(datum['id'])] = {"groups" : [{"words" : proposed_grouping, "theme" : ""} for proposed_grouping in proposed_groupings]}

        # https://en.wikipedia.org/wiki/Assignment_problem
        # https://en.wikipedia.org/wiki/Hungarian_algorithm
        # https://en.wikipedia.org/wiki/Partition_problem
        # https://en.wikipedia.org/wiki/Integer_programming 
        # https://en.wikipedia.org/wiki/Constraint_programming
        # https://www.geeksforgeeks.org/travelling-salesman-problem-using-hungarian-method/#
        
    end_time = time.perf_counter()
    del model
    output = {
        "results" : results,
        "metadata" : {
            "parameters" : args_dict,
            "code_version" : "0.0.4",
              "total_run_time_seconds" : end_time - start_time,
              "write_date" : date.today().strftime("%Y-%m-%d"),
            },
        "raw_similarities" : raw_similarities,
        }
    
    json.dump(output,open(f'results/experiment-{serialized_args}.json','w',encoding='utf-8'))