@echo off
REM Batch script to call evaluate.py with default arguments

python evaluate.py ^
    --output_filename "results_generative.xlsx" ^
    --required_columns model_id total_run_time_seconds ^
    --independent_variables param_count k_shot resolution prompt_version model_family ^
    --dependent_variables mean_accuracy percentage_format_passed ^
    --results_glob "results/0.0.1 results/experiment-*.json" ^
    --connections_file "NYT-Connections-Answers/connections.json" ^
	> materials/generative_table.md 2>&1