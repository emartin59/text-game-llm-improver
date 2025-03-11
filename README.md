# text-game-llm-improver

Code and Models for *"Improving Large Language Models via Self-Play in a Text-Based Game"*, an AI research paper by Eric Martin.

## Overview

This repository contains the implementation of a pioneering framework where Large Language Models (LLMs) autonomously evolve through self-play in a text-based, multi-agent game that blends cooperation and competition. The project demonstrates how LLM-controlled bots can navigate a grid, harvest resources, communicate, and clone, iteratively refining strategies via parameterized rivalry.

Starting with TinyLlama, our proof of concept (POC) yielded 67 enhanced iterations in 47 hours, with the final model achieving an 89.4% win rate against the baseline in 500 randomized games. This framework serves as a testbed for multi-agent dynamics, autonomous decision-making, and emergent intelligence.

## Repository Contents

- **`text-based-game-LLM-trainer.py`**: The main script for running the self-play framework, training LLMs, and simulating the text-based game.
- **`game-model-showdown-1000-games.py`**: A script to run a showdown between models for 1000 games.
- **`LICENSE`**: The MIT License for this project.
- **`README.md`**: This file.

## Models

The trained 8-bit models (including the base TinyLama model and 3 of the 67 iterations derived from TinyLlama) are available for download at the following Google Drive link:

[Download Models](https://drive.google.com/drive/folders/1GkmEAckJUo9fFoUJgW9NRyS39F6umCtO?usp=sharing)

## Getting Started

### Prerequisites

- Python 3.12+
- PyTorch 2.6+
- Transformers library (Hugging Face)
- A CUDA-capable GPU (e.g., NVIDIA RTX with at least 6GB VRAM)
- Conda environment recommended

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/emartin59/text-game-llm-improver.git
   cd text-game-llm-improver

#### Install dependencies:

bash

conda create -n llm-game python=3.12
conda activate llm-game
pip install torch transformers

Optionally download the models from the Google Drive link and place them in a directory of your choice (e.g., models/). These include the base model and some of the models created in the research run, reproduced here for testing and for comparison to models you might create.

[Download Models](https://drive.google.com/drive/folders/1GkmEAckJUo9fFoUJgW9NRyS39F6umCtO?usp=sharing)

#### Running the Code
To run the main self-play training loop using text-based-game-LLM-trainer.py:
bash

python text-based-game-LLM-trainer.py

This script will:
Load the TinyLlama-1.1B-Chat-v1.0 model with 8-bit quantization.

Simulate games on a 7×7 grid (9×9 with boundary walls).

Evolve the LLM through self-play by creating modified clones and evaluating their performance.

Save winning models to directories like New-LLM-Winner-01/.

To run a showdown between two models for 1000 games:
bash

python game-model-showdown-1000-games.py

Hardware Requirements
The proof of concept was run on:
Windows PC

6GB NVIDIA RTX GPU

64GB RAM

AMD Ryzen 7 7700 (3.80 GHz)

## Paper Results
Iterations: 67 improved LLMs over 47 hours.

Performance: The final model achieved an 89.4% win rate against the baseline TinyLlama in 500 randomized games (p < 0.001, t-test).

Emergent Behaviors: While complex tactics like alliances or sabotage did not seem to emerge with TinyLlama, the framework shows potential for strategic optimization with more advanced models.

## Future Work
Scale to larger grids (e.g., 500×500) or 3D environments.

Test with advanced LLMs (e.g., models from OpenAI, xAI, or Anthropic).

Integrate reinforcement learning techniques like PPO or Q-learning for faster convergence.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Thanks to the open-source community for TinyLlama.

Inspired by DeepMind's AlphaZero project.
