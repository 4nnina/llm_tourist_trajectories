# Understanding and Predicting Tourist Behavior through Large Language Models

This repository contains the code, dataset, and experimental results associated with the paper  
**"Understanding and Predicting Tourist Behavior through Large Language Models"**.

The project investigates the use of Large Language Models (LLMs) to interpret tourist trajectories and predict the next Point of Interest (PoI), comparing classical baselines with six open-source LLMs under multiple prompt strategies and anchor configurations.

## Repository Structure

### `dataset/`
This folder contains the dataset used in all experiments.  
It includes the visit records needed to reproduce the baselines and LLM-based evaluations.

### `results.zip`
A compressed archive containing all outputs reported in the paper.

### `src/`
This directory contains the complete implementation of the experimental pipeline.

It includes two main entry points:

- **`main_baselines.py`**  
  Runs all non-LLM baselines (random selection, nearest-PoI, PoI-popularity).

- **`main_LLMs.py`**  
  Executes the LLM-based baselines using:
  - six open-source LLM models ( Llama3.1 8B, Qwen2.5 7B, Qwen2.5 14B, Mixtral 8x7B, Mistral 7B, DeepSeek Coder 33B),
  - both *middle* and *penultimate* anchor strategies,
  - the five incremental prompt designs described in the paper,
  - structured JSON output including predictions and model-generated explanations.

## Requirements

- Python ≥ 3.10  
- Dependencies listed in `requirements.txt`

## Usage

###### Run classical baselines
```bash
python src/main_baselines.py
```

###### Run LLM-based experiments
```bash
python src/main_LLMs.py
```


