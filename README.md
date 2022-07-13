
# Online Decision Transformer

Unofficial implemenation of [Online Decision Transformer](https://arxiv.org/abs/2202.05607) for MuJoCo robotics experiments. 

## Overview

This codebase off original codebase from [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://sites.google.com/berkeley.edu/decision-transformer).

## Installation

As with the original codebase, experiments require MuJoCo.Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f environment.yml
```
### Downloading datasets

Datasets are stored in the `gym/data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```
Alternatively, the environment can be setup using Docker with the attached Dockerfile.

## Implementation Details

There are some small implementation detail differences and other assumed implementation detials that may be different. 

Originally, the policy is parameterized as:
$$\pi_\theta(a_t|s_{-K,t}, g_{-K,t}) = N(\mu_\theta(s_{-K,t}, g_{-K,t}), \Sigma_{\theta}(s_{-K,t}, g_{-K,t}))$$

However, we use $$\mathbf{tanh}(N(\mu_\theta(s_{-K,t}, g_{-K,t}), \Sigma_{\theta}(s_{-K,t}, g_{-K,t})))$$

We first sample from a normal distribution with parameters outputted by the transformer policy and then squash the sampled value to the action space using tanh. 

There is no analytical form for the entropy of this distribution, but we can evaluate the (log) probability of a sample. We use a Monte Carlo estimate over a batch, similarly to the paper, specifically we also sample $k$ actions for each transition within the batch.

Emperically, we find that using tanh leads to better performance, but this change can be ignored by omitting `--stochastic_tanh`. 

## Example usage
We train policies by first performing offline pretraining, and then perform online finetuning seperately. 

### Offline Pretraining

The below command runs offline pretraining with hopper:
```
python experiment.py --env hopper --dataset medium --model_type dt --num_eval_episodes=50 --max_iters=5 --num_steps_per_iter=1000 --stochastic  --use_action_means --learning_rate=1e-4 --embed_dim=512 --weight_decay=5e-4 --K=20 --remove_pos_embs --n_layer=4 --n_head=4 --batch_size=256 --eval_context=5 --stochastic_tanh
```

### Online Learning

```
python experiment.py --env hopper --dataset medium --model_type dt --pretrained_model=./models/hopper/dt_gym-experiment-hopper-medium-506105.pt --stochastic --use_action_means --online_training --eval_context=5 --K=20 --batch_size=256 --num_steps_per_iter=300 --max_iters=200 --num_eval_episodes=50  --device=cuda:2 --target_entropy  --stochastic_tanh
```

Results can be logged by using `--log_to_wandb=True`. We include a pretrained model for hopper. 

## Experimental Results

Experimental results can be found here: 

