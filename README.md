# Towards Solving Kissing Number Problem with Reinforcement Learning and MCTS
Term project for Machine Learning course in Peking University (2023 spring).

The Kissing Number Game is implemented in ```Game.py```. ```Coach.py``` contains the core training loop and ```MCTS.py``` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in ```main.py```, including the dimension, search space, hyperparameters for UCB formula, etc. Neural network architecture is implemented in ```NeuralNet.py```. 

## Installation

An optional first step, which will make everything easier:

```
conda create --name KissingNumber python=3.8.16
conda activate KissingNumber
```

Then, install our project:

```
git clone https://github.com/YK-YoungK/ML_proj_KissingNumber.git
cd ML_proj_KissingNumber
pip install -r requirements.txt
```

To start searching:
```bash
python main.py
```
## Experiments

We present the best results obtained by different models for the kissing number problem in $\mathrm{dim}=3,4,5,6$. In the case of $\mathrm{dim}=5,6$, the RL+MCTS+Knowledge approach utilizes the optimal results from the previous dimension to find the results in the current dimension. Both of our models presented here employ the **''looking one step ahead''** technique. See our report for more details.

![image-20230618163440504](.\fig\Results.png)

