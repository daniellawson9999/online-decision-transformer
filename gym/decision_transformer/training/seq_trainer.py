import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        state_target = torch.clone(states)
        rtg_target = torch.clone(rtg[:,:-1])

        state_preds, action_preds, return_preds, action_log_probs, entropies = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,target_actions=action_target
        )

        act_dim = action_preds.shape[2]
        state_dim = state_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        if action_log_probs != None:
            action_log_probs = action_log_probs.reshape(-1)[attention_mask.reshape(-1) > 0]
        if entropies != None:
            entropies = entropies.reshape(-1)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            state_preds, action_preds, return_preds, None,
            state_target, action_target, rtg_target, None,
            action_log_probs, entropies
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()
        
        
        # Entropy multiplier tuning
        if self.log_entropy_multiplier is not None:
            entropy_multiplier_loss = self.entropy_loss_fn(entropies)
            self.multiplier_optimizer.zero_grad()
            entropy_multiplier_loss.backward()
            self.multiplier_optimizer.step()
            
            entropy_loss = entropy_multiplier_loss.detach().cpu().item()
        else:
            entropy_loss = None

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            if self.log_entropy_multiplier is not None:
                self.diagnostics['training/entropy_multiplier'] = torch.exp(self.log_entropy_multiplier).detach().cpu().item()
                self.diagnostics['training/entropy'] = torch.mean(entropies).item()
        return loss.detach().cpu().item(), entropy_loss
