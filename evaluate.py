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
    id_to_idx_mapping = {str(nyt_connections_datum['id']) : item_idx for item_idx,nyt_connections_datum in enumerate(nyt_connections_data)}
    for result_id, result in results.items():
        item_idx = id_to_idx_mapping[result_id]
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

def parser_args():
    parser = argparse.ArgumentParser(description="Process experiment data for analysis.")
    
    parser.add_argument(
        "--output_filename", 
        type=str, 
        default="results_structured.xlsx", 
        help="Output file name (default: results_structured.xlsx)"
    )
    parser.add_argument(
        "--required_columns", 
        type=str, 
        nargs='+', 
        default=['model_id', 'total_run_time_seconds'], 
        help="List of required columns (default: ['model_id', 'total_run_time_seconds'])"
    )
    parser.add_argument(
        "--independent_variables", 
        type=str, 
        nargs='+', 
        default=['param_count', 'prompt_k', 'resolution', 'prompt_version', 'model_family'], 
        help="List of independent variables (default: ['param_count', 'prompt_k', 'resolution', 'prompt_version', 'model_family'])"
    )
    parser.add_argument(
        "--dependent_variables", 
        type=str, 
        nargs='+', 
        default=['mean_accuracy'], 
        help="List of dependent variables (default: ['mean_accuracy'])"
    )
    parser.add_argument(
        "--results_glob", 
        type=str, 
        default='results/experiment-*.json', 
        help="Glob pattern for results files (default: 'results/experiment-*.json')"
    )
    parser.add_argument(
        "--connections_file", 
        type=str, 
        default="NYT-Connections-Answers/connections.json", 
        help="Path to the connections file (default: 'NYT-Connections-Answers/connections.json')"
    )

    # Parse arguments
    args = parser.parse_args()

    return args
    
if __name__ == "__main__":
    from glob import glob
    
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import StandardScaler
    from tqdm.auto import tqdm
    
    
    model_to_param_count = {'meta-llama/Meta-Llama-3.1-8B-Instruct' : 8,
    'meta-llama/Llama-3.2-3B-Instruct' : 3,
    'Qwen/Qwen2.5-3B-Instruct'  : 3,
    'meta-llama/Llama-3.2-1B-Instruct' : 1,
    'Qwen/Qwen2.5-14B-Instruct' : 14,
    'Qwen/Qwen2.5-7B-Instruct' : 7,
    'Qwen/Qwen2.5-1.5B-Instruct' : 1.5}

    args = parser_args()
    output_filename = args.output_filename
    required_columns = args.required_columns
    independent_variables = args.independent_variables
    dependent_variables = args.dependent_variables
    results_glob = args.results_glob
    connections_file = args.connections_file
    
    output_columns = required_columns + independent_variables + dependent_variables
    
    nyt_connections_data = json.load(open(connections_file,'r'))
    global_results = []
    for filename in tqdm(glob(results_glob)):
        model_results = json.load(open(filename,'r',encoding='utf-8'))
        model_results['metrics'] = calculate_statistics(model_results['results'],nyt_connections_data)
        global_results.append(model_results)
    
    global_results = [format_result(global_result) for global_result in global_results if global_result['results']]
    # TODO: FLATTEN GLOBAL RESULTS ONE LEVEL
    global_results_df = pd.DataFrame(global_results)
    #global_results_df = global_results_df[global_results_df['resolution']]
    global_results_df['param_count'] = global_results_df['model_id'].map(model_to_param_count)
    global_results_df['model_family'] = global_results_df['model_id'].apply(lambda x: 'Qwen' if 'Qwen' in x else 'Llama3')
    print(global_results_df[output_columns].sort_values(required_columns+independent_variables).to_markdown(index=False))

    for dependent_variable in dependent_variables:
        for independent_variable in independent_variables:
            result = spearmanr(global_results_df[independent_variable], global_results_df[dependent_variable])
            print(independent_variable,dependent_variable,result)

    for dependent_variable in dependent_variables:
        # TODO: FIND A WAY OF CONVERTING STRINGS/OBJECTS BEFORE SCALER
        independent_variable_items = global_results_df[independent_variables]
        # Separate non-numeric columns
        non_numeric_columns = independent_variable_items.select_dtypes(include=['object']).columns

        # Encode non-numeric cols
        ordinal_encoder = OrdinalEncoder()
        independent_variable_items[non_numeric_columns] = ordinal_encoder.fit_transform(independent_variable_items[non_numeric_columns])

        dependent_variable_items = global_results_df[dependent_variable].values.reshape(-1, 1)

        # Standardize features for better interpretation
        scaler = StandardScaler()
        scaled_independent_variable_items = scaler.fit_transform(independent_variable_items)
        scaled_dependent_variable_items = scaler.fit_transform(dependent_variable_items).flatten()
        
        # Fit linear regression model to predict 'mean_accuracy'
        reg_accuracy = LinearRegression()
        reg_accuracy.fit(scaled_independent_variable_items, scaled_dependent_variable_items)
        print("Coefficients for 'mean_accuracy':", dict(zip(independent_variable_items.columns, reg_accuracy.coef_)))
    
    global_results_df[output_columns].to_excel(output_filename,index=False)