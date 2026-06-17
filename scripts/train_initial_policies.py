import os
import json
import argparse
import shutil
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from utils import create_env, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", type=str, choices=["discrete", "continuous"], default="discrete")
    parser.add_argument("--frame_stack", type=int, default=1)

    # [추가] 기존 저장 모델이 있으면 이어서 학습
    parser.add_argument("--resume", action="store_true")

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
    env = create_env(
        env_config['env_name'],
        env_config['env_params'],
        args.env_type,
        frame_stack=args.frame_stack,
        seed=seed
    )

    policy_kwargs = dict(net_arch=env_config.get('net_arch', [256, 256]))
    
    tb_log_dir = "./tb_logs"
    os.makedirs(tb_log_dir, exist_ok=True)

    # =====================================================================
    # 1. Suboptimal Policy 학습
    # =====================================================================

    # [추가] Suboptimal Checkpoint 저장
    suboptimal_checkpoint_dir = f"./{args.env_type}_suboptimal_checkpoints"
    os.makedirs(suboptimal_checkpoint_dir, exist_ok=True)

    suboptimal_checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=suboptimal_checkpoint_dir,
        name_prefix="suboptimal_step"
    )

    # [추가] resume이면 기존 suboptimal_policy 로드
    if args.resume and os.path.exists(env_config['paths']['suboptimal_policy']):
        print(f"[RESUME] 기존 Suboptimal Policy 로드: {env_config['paths']['suboptimal_policy']}")

        if args.env_type == "discrete":
            suboptimal_model = DQN.load(
                env_config['paths']['suboptimal_policy'],
                env=env,
                device=device,
                tensorboard_log=tb_log_dir
            )
        else:
            suboptimal_model = SAC.load(
                env_config['paths']['suboptimal_policy'],
                env=env,
                device=device,
                tensorboard_log=tb_log_dir
            )

    else:
        print("[NEW] 새 Suboptimal Policy 생성")

        if args.env_type == "discrete":
            suboptimal_model = DQN(
                env_config['policy_type'], 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                tensorboard_log=tb_log_dir,
                buffer_size=100000, 
                learning_starts=5000, 
                optimize_memory_usage=True,
                replay_buffer_kwargs=dict(handle_timeout_termination=False),
                device=device,
                seed=seed
            )
        else:
            suboptimal_model = SAC(
                env_config['policy_type'], 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                tensorboard_log=tb_log_dir,
                buffer_size=100000, 
                batch_size=256, 
                optimize_memory_usage=True, 
                replay_buffer_kwargs=dict(handle_timeout_termination=False),
                device=device,
                seed=seed
            )

    suboptimal_model.learn(
        total_timesteps=rl_config['suboptimal_steps'],
        reset_num_timesteps=not args.resume,
        progress_bar=True,
        log_interval=1,
        callback=suboptimal_checkpoint_callback
    )

    suboptimal_model.save(env_config['paths']['suboptimal_policy'])
    print(f"Suboptimal Policy 저장 완료: {env_config['paths']['suboptimal_policy']}")

    # =====================================================================
    # 2. Expert Policy 학습
    # =====================================================================
    print("\n--- Expert Policy 학습 시작 ---")
    eval_env = create_env(
        env_config['env_name'],
        env_config['env_params'],
        args.env_type,
        frame_stack=args.frame_stack,
        seed=seed + 100
    )
    
    checkpoint_dir = f"./{args.env_type}_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoint_dir,
        name_prefix="expert_step"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./tmp_best",
        log_path=None,
        eval_freq=50000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # [수정] resume이고 expert_policy가 이미 있으면 expert도 이어서 학습
    if args.resume and os.path.exists(env_config['paths']['expert_policy']):
        print(f"[RESUME] 기존 Expert Policy 로드: {env_config['paths']['expert_policy']}")

        if args.env_type == "discrete":
            expert_model = DQN.load(
                env_config['paths']['expert_policy'],
                env=env,
                device=device,
                tensorboard_log=tb_log_dir
            )
        else:
            expert_model = SAC.load(
                env_config['paths']['expert_policy'],
                env=env,
                device=device,
                tensorboard_log=tb_log_dir
            )

        # resume일 때는 config의 expert_steps만큼 추가 학습
        full_steps = rl_config['expert_steps']

    else:
        print("[NEW] Suboptimal Policy에서 Expert 학습 시작")

        if args.env_type == "discrete":
            expert_model = DQN.load(
                env_config['paths']['suboptimal_policy'],
                env=env,
                device=device,
                tensorboard_log=tb_log_dir
            )
        else:
            expert_model = SAC.load(
                env_config['paths']['suboptimal_policy'],
                env=env,
                device=device,
                tensorboard_log=tb_log_dir
            )

        full_steps = rl_config['expert_steps'] - rl_config['suboptimal_steps']

    if full_steps <= 0:
        print(f"Expert 추가 학습 step이 {full_steps}입니다. Expert 학습을 건너뜁니다.")
        expert_model.save(env_config['paths']['expert_policy'])
    else:
        expert_model.learn(
            total_timesteps=full_steps,
            reset_num_timesteps=not args.resume,
            callback=callback_list,
            progress_bar=True,
            log_interval=1
        )
        
        best_model_path = os.path.join("./tmp_best", "best_model.zip")

        if os.path.exists(best_model_path):
            shutil.move(best_model_path, env_config['paths']['expert_policy'])
            print("최고 성능의 Best Expert Policy로 최종 저장 완료!")
        else:
            expert_model.save(env_config['paths']['expert_policy'])
            print(f"Expert Policy 저장 완료: {env_config['paths']['expert_policy']}")
        
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()