# List of hyperparameter options
model_ids = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]
prompt_versions = ["themed","default"]
k_shot_options = [0, 1, 3, 5]
sample_size = 50  # Fixed sample size
resolutions = [4,8,16] #,8,16]

script_name = "group_preds_poc.py"

# Executes a gridsearch for hyperparameters above and writes to a batch file
def generate_experiment_bat(output_filename):
    with open(output_filename, "w") as bat_file:
        bat_file.write("@echo off\n")
        for model_id in model_ids:
            for resolution in resolutions:
                for prompt_version in prompt_versions:
                    for k_shot_option in k_shot_options:
                        # Create the command string
                        command = f"python {script_name} --model_id {model_id} --prompt_version {prompt_version} --prompt_k {k_shot_option} --sample_size {sample_size} --resolution {resolution}\n"
                        bat_file.write(command)

if __name__ == "__main__":
    generate_experiment_bat("structured_grid_search.bat")