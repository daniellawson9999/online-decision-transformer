TODO, update
Commands:

Hopper:
Pretraining:

python experiment.py --env hopper --dataset medium --model_type dt --num_eval_episodes=50 --max_iters=5 --num_steps_per_iter=1000 --stochastic  --use_action_means --learning_rate=1e-4 --embed_dim=512 --weight_decay=5e-4 --K=20 --remove_pos_embs --n_layer=4 --n_head=4 --batch_size=256 --eval_context=5 --device=cuda:2 --log_to_wandb=True --stochastic_tanh

Online finetuning:
python experiment.py --env hopper --dataset medium --model_type dt --pretrained_model=./models/hopper/dt_gym-experiment-hopper-medium-506105.pt --stochastic --use_action_means --online_training --eval_context=5 --K=20 --batch_size=256 --num_steps_per_iter=300 --max_iters=200 --num_eval_episodes=50 --stochastic_tanh --device=cuda:2 --log_to_wandb=True 

python experiment.py --env hopper --dataset medium --model_type dt --pretrained_model=./models/hopper/dt_gym-experiment-hopper-medium-506105.pt --stochastic --use_action_means --online_training --eval_context=5 --K=20 --batch_size=256 --num_steps_per_iter=300 --max_iters=200 --num_eval_episodes=50  --device=cuda:2 --target_entropy --log_to_wandb=True --stochastic_tanh


Walker2D:
#Fix, this is wrong
pretraining:
python experiment.py --env walker2d --dataset medium --model_type dt --num_eval_episodes=50 --max_iters=5 --num_steps_per_iter=2000 --stochastic  --use_action_means --learning_rate=1e-3 --embed_dim=512 --weight_decay=1e-3 --K=20 --remove_pos_embs --n_layer=4 --n_head=4 --batch_size=256 --eval_context=5 --stochastic_tanh --device=cuda:2 --log_to_wandb=True



python experiment.py --env walker2d --dataset medium --model_type dt --pretrained_model=./models/walker2d/dt_gym-experiment-walker2d-medium-763104.pt --stochastic --use_action_means --online_training --eval_context=5 --K=20 --batch_size=256 --num_steps_per_iter=300 --max_iters=200 --num_eval_episodes=50 --learning_rate=1e-3 --weight_decay=1e-3 --device=cuda:2 --log_to_wandb=True --target_entropy --stochastic_tanh



Model-based testing:
 python experiment.py --env halfcheetah --dataset medium --model_type dt --num_eval_episodes=10 --max_iters=1 --num_steps_per_iter=0 --stochastic --device=cuda:1 --use_model --pretrained_model=./models/halfcheetah/dt_gym-experiment-halfcheetah-medium-268755.pt --pretrained_mode=static --use_action_means --plan_horizon=25 --number_rollouts=10