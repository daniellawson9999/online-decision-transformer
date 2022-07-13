import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            stochastic=False,
            log_std_min=-20,
            log_std_max=2,
            remove_pos_embs=False,
            stochastic_tanh=False,
            approximate_entropy_samples=1000,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        # Settings from stochastic actions
        self.stochastic = stochastic
        self.log_std_min=log_std_min
        self.log_std_max=log_std_max
        self.stochastic_tanh=stochastic_tanh
        self.approximate_entropy_samples=approximate_entropy_samples



        self.remove_pos_embs = remove_pos_embs
        if not remove_pos_embs:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)

        if stochastic:
            self.predict_action_mean = nn.Sequential(
                nn.Linear(hidden_size, self.act_dim),
            )
            self.predict_action_logstd = nn.Sequential(
                nn.Linear(hidden_size, self.act_dim),
            )
        else:
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, target_actions=None, use_means=False):

        batch_size, seq_length = states.shape[0], states.shape[1]

        transition_size = 3

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        
        # Optionally can remove, may be better for certain domains if order can be inferred by return seq
        if not self.remove_pos_embs:
            time_embeddings = self.embed_timestep(timesteps)

            # time embeddings are treated similar to positional embeddings
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings
            returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        embeddings = (returns_embeddings, state_embeddings, action_embeddings)
        stacked_inputs = torch.stack(
            embeddings, dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, transition_size*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        attention_masks = (attention_mask, attention_mask, attention_mask)

        stacked_attention_mask = torch.stack(
            attention_masks, dim=1
        ).permute(0, 2, 1).reshape(batch_size, transition_size*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            use_cache=False
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        # or rewards (3)
        x = x.reshape(batch_size, seq_length, transition_size, self.hidden_size).permute(0, 2, 1, 3)

        state_reps = x[:,1]
        action_reps = x[:,2] 

        # get predictions
        return_preds = self.predict_return(action_reps)  # predict next return given state and action
        state_preds = self.predict_state(action_reps)    # predict next state given state and action
        

        action_log_probs = None
        entropies = None
        if self.stochastic:
            
            means = self.predict_action_mean(state_reps)
            log_stds = self.predict_action_logstd(state_reps)

            # Bound log of standard deviations
            log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
            stds = torch.exp(log_stds)

            #action_distributions = TransformedDistribution(Normal(means, stds), TanhTransform(cache_size=1))
            #action_distributions = Normal(means, stds)
            
            if self.stochastic_tanh:
                action_distributions = Independent(TransformedDistribution(Normal(means, stds), TanhTransform(cache_size=1)),1)
            else:
                action_distributions = Independent(Normal(means, stds),1)
            # Sample from distribution or predict mean
            if use_means:
                if self.stochastic_tanh:
                    action_preds = torch.tanh(action_distributions.base_dist.base_dist.mean)
                else:
                    action_preds = action_distributions.mean
            else:
                action_preds = action_distributions.rsample()

            if target_actions != None:
                # Clamp target actions to prevent nans
                eps = torch.finfo(target_actions.dtype).eps
                target_actions = torch.clamp(target_actions, -1+eps, 1-eps)
                action_log_probs = action_distributions.log_prob(target_actions)       
                #entropies = action_distributions.base_dist.entropy()
                if self.stochastic_tanh:
                    entropies = -action_distributions.log_prob(action_distributions.rsample(sample_shape=torch.Size([self.approximate_entropy_samples]))).mean(dim=0)
                else:
                    entropies = action_distributions.entropy()
                

        else:
            action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds, action_log_probs, entropies

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, use_means=False, custom_max_length=None,**kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        max_length = self.max_length
        if custom_max_length is not None:
            max_length = custom_max_length
        if max_length is not None:
            states = states[:,-max_length:]
            actions = actions[:,-max_length:]
            returns_to_go = returns_to_go[:,-max_length:]
            timesteps = timesteps[:,-max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        state_preds, action_preds, return_preds, _, _ = self.forward(
            states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask, use_means=use_means, **kwargs)
        return action_preds[0,-1]

