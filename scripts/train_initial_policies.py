import os
import json
import argparse
import shutil
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
from utils import create_env, get_device

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

    env_config = config[args.env_type]
    rl_config = config['rl_training']
    
    os.makedirs("./discrete_models", exist_ok=True)
    os.makedirs("./continuous_models", exist_ok=True)

    print(f"[{args.env_type.upper()}] 기초 모델 학습 시작 (Device: {device}, Seed: {seed})")
    env = create_env(env_config['env_name'], env_config['env_params'], args.env_type, frame_stack=args.frame_stack, seed=seed)

    policy_kwargs = dict(net_arch=env_config.get('net_arch', [256, 256]))
    
    # 1. Suboptimal Policy 학습 (버퍼 축소 및 메모리 최적화 적용)
    if args.env_type == "discrete":
        suboptimal_model = DQN(env_config['policy_type'], 
                               env, 
                               policy_kwargs=policy_kwargs, 
                               verbose=1, 
                               buffer_size=20000, 
                               learning_starts=5000, 
                               optimize_memory_usage=True,
                               replay_buffer_kwargs=dict(handle_timeout_termination=False),
                               device=device, seed=seed)
    else:
        suboptimal_model = SAC(env_config['policy_type'], 
                               env, 
                               policy_kwargs=policy_kwargs, 
                               verbose=1, 
                               buffer_size=20000, 
                               batch_size=256, 
                               optimize_memory_usage=True, 
                               replay_buffer_kwargs=dict(handle_timeout_termination=False),
                               device=device, seed=seed)
                               
    suboptimal_model.learn(total_timesteps=rl_config['suboptimal_steps'], progress_bar=True)
    suboptimal_model.save(env_config['paths']['suboptimal_policy'])

    # 2. Expert Policy 학습 (Eval Callback 적용)
    print("\n--- Expert Policy 학습 시작 ---")
    eval_env = create_env(env_config['env_name'], env_config['env_params'], args.env_type, frame_stack=args.frame_stack, seed=seed+100)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./tmp_best", log_path=None, eval_freq=50000, n_eval_episodes=5, deterministic=True, render=False)
    
    expert_model = DQN.load(env_config['paths']['suboptimal_policy'], env, device=device) if args.env_type == "discrete" \
                   else SAC.load(env_config['paths']['suboptimal_policy'], env, device=device)
        
    full_steps = rl_config['expert_steps'] - rl_config['suboptimal_steps']
    expert_model.learn(total_timesteps=full_steps, reset_num_timesteps=False, callback=eval_callback, progress_bar=True)
    
    best_model_path = os.path.join("./tmp_best", "best_model.zip")
    if os.path.exists(best_model_path):
        shutil.move(best_model_path, env_config['paths']['expert_policy'])
        print("최고 성능의 Best Expert Policy로 저장 완료!")
    else:
        expert_model.save(env_config['paths']['expert_policy'])
        
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()