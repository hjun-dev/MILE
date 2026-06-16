import json
import argparse
from stable_baselines3 import DQN, SAC
import os
from stable_baselines3.common.utils import set_random_seed
from utils import create_env, get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", type=str, choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--round", type=int, default=5)
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        config = json.load(f)

    device = get_device()
    seed = config.get("global_seed", 42)
    set_random_seed(seed)

    env_config = config[args.env_type]
    base_path = env_config['paths']['final_trained_policy']
    basename, ext = os.path.splitext(base_path)
    model_path = f"{basename}_round_{args.round}{ext}"
    
    print(f"[{args.env_type.upper()} | Stack: {args.frame_stack}] 모델 렌더링 테스트: {model_path} (Device: {device})")
    env = create_env(env_config['env_name'], env_config['env_params'], args.env_type, frame_stack=args.frame_stack, render_mode="human", seed=seed)
    
    model = DQN.load(model_path, env=env, device=device) if args.env_type == "discrete" else SAC.load(model_path, env=env, device=device)

    for episode in range(3):
        obs = env.reset() 
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            done = dones[0]
        print(f"Episode {episode + 1} - Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()