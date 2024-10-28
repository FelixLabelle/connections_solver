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
use_structured_prediction_options = [True, False]
k_shot_options = [0, 1, 3, 5]
sample_size = 50  # Fixed sample size
resolutions = [4] #,8,16]

script_name = "connections_solver.py"

# Executes a gridsearch for hyperparameters above and writes to a batch file
def generate_experiment_bat(output_filename):
    with open(output_filename, "w") as bat_file:
        bat_file.write("@echo off\n")
        for resolution in resolutions:
            for model_id in model_ids:
                for use_structured_prediction in use_structured_prediction_options:
                    for k_shot in k_shot_options:
                        # Create the command string
                        structured_flag = "--use_structured_prediction" if use_structured_prediction else ""
                        command = f"python {script_name} --model_id {model_id} {structured_flag} --k_shot {k_shot} --sample_size {sample_size} --resolution {resolution}\n"
                        bat_file.write(command)

if __name__ == "__main__":
    generate_experiment_bat("grid_search_v3.bat")