import argparse
import json
import re
import statistics

def generate_validation_regex(words):
    # I'm not proud of how this function looks, there is likely a nicer way of doing this, but here we are
    qouted_words = [f"\"{word}\"" for word in words]
    words_regex = f"({'|'.join(qouted_words)})"

    backreference_terms = [""] + ["(?!{})".format("".join(["\\{}".format(j) for j in range(1, i+1)])) for i in range(1, 16)]

    regex_str = r'\[\s*{{\s*"words"\s*:\s*\[\s*{words}\s*,\s*{back1}{words}\s*,\s*{back2}{words}\s*,\s*{back3}{words}\s*\]\s*,\s*"theme"\s*:\s*"[^"]*"\s*}}\s*,\s*{{\s*"words"\s*:\s*\[\s*{back4}{words}\s*,\s*{back5}{words}\s*,\s*{back6}{words}\s*,\s*{back7}{words}\s*\]\s*,\s*"theme"\s*:\s*"[^"]*"\s*}}\s*,\s*{{\s*"words"\s*:\s*\[\s*{back8}{words}\s*,\s*{back9}{words}\s*,\s*{back10}{words}\s*,\s*{back11}{words}\s*\]\s*,\s*"theme"\s*:\s*"[^"]*"\s*}}\s*,\s*{{\s*"words"\s*:\s*\[\s*{back12}{words}\s*,\s*{back13}{words}\s*,\s*{back14}{words}\s*,\s*{back15}{words}\s*\]\s*,\s*"theme"\s*:\s*"[^"]*"\s*}}\s*\]'.format(words=words_regex,
        back1=backreference_terms[1], back2=backreference_terms[2], back3=backreference_terms[3],
        back4=backreference_terms[4], back5=backreference_terms[5], back6=backreference_terms[6], back7=backreference_terms[7],
        back8=backreference_terms[8], back9=backreference_terms[9], back10=backreference_terms[10], back11=backreference_terms[11],
        back12=backreference_terms[12], back13=backreference_terms[13], back14=backreference_terms[14], back15=backreference_terms[15]
    )

    return regex_str.strip()
    
def calculate_statistics(results,nyt_connections_data):
    vals = []
    stats = {}
    format_types = []
    for result_id, result in results.items():
        item_idx = int(result_id) - 1
        if isinstance(result,dict):
            groups = result['groups']
            pred = {frozenset(group["words"]) for group in groups}
            gt = {frozenset(answers['members']) for answers in nyt_connections_data[item_idx]['answers']}
            gt_words = [word for fs in gt for word in fs]
            regex_str = generate_validation_regex(gt_words)
            format_matched = re.match(regex_str,json.dumps(groups))
            if format_matched:
                format_types.append("Valid")
            else:
                format_types.append("Incorrect Format")
                
            num_matches = len(pred.intersection(gt))
            if num_matches > 0:
                pass
            
        else:
            format_types.append("Incorrect Type")
            num_matches = 0
        vals.append(num_matches/4)
        # Calculate overlaps
        # Track stats on difficulty
    if results:
        stats['mean_accuracy'] = statistics.mean(vals)
        stats['accuracy'] = vals
        stats['format_info'] = format_types
        stats['percentage_format_passed'] = sum([format_type == "Valid" for format_type in format_types])/len(format_types)
    return stats

def flatten_dict(dict_to_flatten,leading_text="",depth=0,max_depth=-1):
    flattened_dict = {}
    for key,val in dict_to_flatten.items():
        if not isinstance(val,dict) or (depth > max_depth and max_depth!=-1):
            flattened_dict[f"{leading_text}{key}"] = val
        else:
            flattened_dict.update(flatten_dict(val,leading_text=f"",depth=depth+1,max_depth=max_depth))
    return flattened_dict
    
def format_result(model_results):
   metadata_dict = flatten_dict(model_results['metadata'],max_depth=1)
   results_dict = flatten_dict(model_results['metrics'],max_depth=1)
   metadata_dict.update(results_dict)
   return metadata_dict
 
if __name__ == "__main__":
    from glob import glob
    
    import pandas as pd
    from tqdm.auto import tqdm
    
    model_to_param_count = {'meta-llama/Meta-Llama-3.1-8B-Instruct' : 8,
    'meta-llama/Llama-3.2-3B-Instruct' : 3,
    'Qwen/Qwen2.5-3B-Instruct'  : 3,
    'meta-llama/Llama-3.2-1B-Instruct' : 1,
    'Qwen/Qwen2.5-14B-Instruct' : 14,
    'Qwen/Qwen2.5-1.5B-Instruct' : 1.5}

    output_columns = ['model_id', 'use_structured_prediction', 'k_shot',
           'mean_accuracy','percentage_format_passed']
       
    nyt_connections_data = json.load(open("NYT-Connections-Answers/connections.json",'r'))
    global_results = []
    for filename in tqdm(glob(f'results/0.0.1 results/experiment-*.json')):
        model_results = json.load(open(filename,'r',encoding='utf-8'))
        model_results['metrics'] = calculate_statistics(model_results['results'],nyt_connections_data)
        global_results.append(model_results)
    
    global_results = [format_result(global_result) for global_result in global_results if global_result['results']]
    # TODO: FLATTEN GLOBAL RESULTS ONE LEVEL
    global_results_df = pd.DataFrame(global_results)
    global_results_df = global_results_df[global_results_df['resolution'] == 4]
    global_results_df['param_count'] = global_results_df['model_id'].map(model_to_param_count)
    #import pdb;pdb.set_trace()
    from scipy.stats import spearmanr
    size_performance_correlation = spearmanr(global_results_df['param_count'], global_results_df['mean_accuracy'])
    kshot_performance_correlation = spearmanr(global_results_df['k_shot'], global_results_df['mean_accuracy'])
    structure_performance_correlation = spearmanr(global_results_df['use_structured_prediction'], global_results_df['mean_accuracy'])
    structure_validity_correlation = spearmanr(global_results_df['use_structured_prediction'], global_results_df['percentage_format_passed'])
    size_validity_correlation = spearmanr(global_results_df['param_count'], global_results_df['percentage_format_passed'])
    kshot_validity_correlation = spearmanr(global_results_df['k_shot'], global_results_df['percentage_format_passed'])
    print(global_results_df[output_columns].to_markdown(index=False))
    global_results_df.to_excel("results.xlsx",index=False)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    experiment_variables = global_results_df[['param_count', 'k_shot', 'use_structured_prediction']]
    y_accuracy = global_results_df['mean_accuracy'].values.reshape(-1, 1)
    y_validity = global_results_df['percentage_format_passed'].values.reshape(-1, 1)

    # Standardize features for better interpretation
    scaler = StandardScaler()
    scaled_experiment_variables = scaler.fit_transform(experiment_variables)
    scaled_accuracy = scaler.fit_transform(y_accuracy).flatten()
    scaled_validity = scaler.fit_transform(y_validity).flatten()
    # Fit linear regression model to predict 'mean_accuracy'
    reg_accuracy = LinearRegression()
    reg_accuracy.fit(scaled_experiment_variables, scaled_accuracy)
    print("Coefficients for 'mean_accuracy':", dict(zip(experiment_variables.columns, reg_accuracy.coef_)))

    # Fit linear regression model to predict 'percentage_format_passed'
    reg_validity = LinearRegression()
    reg_validity.fit(scaled_experiment_variables, scaled_validity)
    print("Coefficients for 'percentage_format_passed':", dict(zip(experiment_variables.columns, reg_validity.coef_)))