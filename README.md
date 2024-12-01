# README

## Overview

Repo containing "NYT Connections" solvers associated with the following blog post:  
[Fun with Words: NYT Connections](https://felixlabelle.com/2024/10/30/fun-with-words-nyt-connections.html)

## Getting Started

Clone the repo and initialize submodules properly using the following commands:

```bash
git clone --recurse-submodules <repo-url>
```

The environment is defined in `env.yml` and was built using Conda on Windows 11. Compatibility with other operating systems has not been tested and may require adjustments.

## Running the Code

### Repo Structure

**Top-Level Files**  
- `analyze_binary_performance.py`: Analyzes performance metrics for binary experiments.  
- `config.py`: Configuration settings for the solvers and experiments.  
- `data_analysis.py`: Tools for analyzing and visualizing data from experiments.  
- `env.yml`: Environment file to set up the Conda environment.  
- `evaluate.py`: Evaluates model performance and generates summarized outputs.  
- `evaluate_generative.bat`: Batch file for evaluating generative solver experiments.  
- `evaluate_structured.bat`: Batch file for evaluating structured solver experiments.  
- `generate_generative_experiments.py`: Script for setting up generative solver experiments.  
- `generate_structured_experiments.py`: Script for setting up structured solver experiments.  
- `generation_structured_prediction_compared.py`: Compares predictions from generative and structured approaches.  
- `generative_connections_solver.py`: Implementation of the generative solver for NYT Connections.  
- `structured_connections_solver.py`: Implementation of the structured solver for NYT Connections.  
- `utils.py`: Helper functions used across the project.

**Folders**  
- `analysis_results`: Contains output results (Excel files and tables) from experiments.  
- `NYT-Connections-Answers`: Submodule containing answer data for NYT Connections, with an additional workflow for updates.  

### Replicating the Experiments

1. **Run Experiments**  
   Use the appropriate `grid_search` batch file to run all experiments:  
   ```bash
   structured_grid_search.bat
   generative_grid_search.bat
   ```

   These experiments may require a GPU with at least 16GB of VRAM for optimal performance.

2. **Analyze Results**  
   Use the corresponding `evaluate_*.bat` file to analyze the results:  
   ```bash
   evaluate_structured.bat
   evaluate_generative.bat
   ```

   These scripts generate Excel files (`.xlsx`) containing performance metrics and calculate correlations between independent and dependent variables.
