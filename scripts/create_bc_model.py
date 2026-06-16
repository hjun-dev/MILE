import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from imitation.data import rollout
from imitation.algorithms.bc import BC
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.utils import set_random_seed

from utils import create_env, MentalModelPolicy, get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", type=str, choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--frame_stack", type=int, default=1)
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        config = json.load(f)

    device = get_device()
    seed = config.get("global_seed", 42)
    set_random_seed(seed)
    rng = np.random.default_rng(seed)

    env_config = config[args.env_type]
    bc_config = config['bc_training']
    net_arch = env_config.get('net_arch', [256, 256])

    venv = create_env(env_config['env_name'], env_config['env_params'], args.env_type, frame_stack=args.frame_stack, for_imitation=True, seed=seed)
    print(f"[{args.env_type.upper()}] BC 학습 시작 (Device: {device})")
    
    if args.env_type == "discrete":
        suboptimal_policy = DQN.load(env_config['paths']['suboptimal_policy'], venv, device=device)
        rollouts = rollout.rollout(suboptimal_policy, venv, rollout.make_min_episodes(bc_config['num_rollouts']), rng=rng)
        transitions = rollout.flatten_trajectories(rollouts)
        
        train_loader = DataLoader(TensorDataset(torch.tensor(transitions.obs, dtype=torch.float32), 
                                                torch.tensor(transitions.acts, dtype=torch.long)), batch_size=64, shuffle=True)
        
        mental_model = MentalModelPolicy(venv.observation_space.shape, venv.action_space.n, net_arch).to(device)
        optimizer = optim.Adam(mental_model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(bc_config['num_epochs']):
            for states, actions in train_loader:
                states, actions = states.to(device), actions.to(device)
                optimizer.zero_grad()
                loss = loss_fn(mental_model(states), actions)
                loss.backward()
                optimizer.step()
        torch.save(mental_model.state_dict(), env_config['paths']['initial_mental_model'])

    else:
        suboptimal_policy = SAC.load(env_config['paths']['suboptimal_policy'], venv, device=device)
        rollouts = rollout.rollout(suboptimal_policy, venv, rollout.make_min_episodes(bc_config['num_rollouts']), rng=rng)
        transitions = rollout.flatten_trajectories(rollouts)
        
        dummy_sac = SAC(env_config['policy_type'], venv, policy_kwargs=dict(net_arch=net_arch), device=device, seed=seed)
        bc_trainer = BC(observation_space=venv.observation_space, action_space=venv.action_space,
                        policy=dummy_sac.policy, rng=rng, demonstrations=transitions, device=device)
        bc_trainer.train(n_epochs=bc_config['num_epochs'])
        
        dummy_sac.policy = bc_trainer.policy
        dummy_sac.save(env_config['paths']['initial_mental_model'])
        
    print(f"[{args.env_type.upper()}] BC Mental Model 저장 완료!")

if __name__ == "__main__":
    main()