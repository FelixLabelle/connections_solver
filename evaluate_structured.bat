@echo off
REM Batch script to call evaluate.py with default arguments

python evaluate.py ^
    --output_filename "results_structured.xlsx" ^
    --required_columns model_id total_run_time_seconds ^
    --independent_variables param_count prompt_k resolution prompt_version model_family ^
    --dependent_variables mean_accuracy ^
    --results_glob "results/experiment-*.json" ^
    --connections_file "NYT-Connections-Answers/connections.json" ^
	> materials/structured_table.md 2>&1