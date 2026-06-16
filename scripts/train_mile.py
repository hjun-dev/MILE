import json
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.distributions import TanhBijector
from stable_baselines3.common.utils import set_random_seed

from utils import DictDataset, MentalModelPolicy, create_env, get_device
from computational_model import computational_intervention_model, continuous_loss_fn

'''
[데이터 수집 핵심 로직 및 오류 수정 사항]
이 함수는 환경과 상호작용하며 로봇의 행동(rollout)과 가상 인간(expert)의 개입 여부를 수집합니다.

* ⚠️ 원본 코드(Lira lab) 대비 개선된 3대 핵심 포인트:
1. Ground-Truth Mental Model 분리:
   원본은 가상 인간이 개입할 때 '로봇이 학습 중인 변동하는 멘탈 모델'을 넘겨서 판단하게 하는 치명적 오류가 있었습니다.
   여기서는 인간의 확고한 기준점인 `gt_mental_model`(학습되지 않고 고정됨)을 주입하여, 
   인간은 흔들림 없이 개입하고 로봇(learning_mental_model)은 이를 추종하게 만듭니다.
2. 메모리 폭발(OOM) 완벽 차단:
   환경과 수천 번 상호작용하는 이 함수에 `with torch.no_grad():`와 `.eval()`이 누락되면,
   Pytorch 연산 그래프(Gradient Map)가 RAM과 GPU를 가득 채워 라운드 1~2에서 무조건 뻗어버립니다. 이를 완벽히 방어했습니다.
3. SAC Tanh 역변환 시 클리핑(Clipping):
   SAC 알고리즘 특성상 행동값이 [-1, 1]에 갇히는데, TanhBijector.inverse를 통과할 때 값이 1.0이면 무한대(Inf)가 발생합니다.
   .clamp(-0.99999, 0.99999)를 적용해 수학적 붕괴를 원천 차단했습니다.
'''
def collect_synthetic_data(venv, rollout_policy_full, expert_policy_net, gt_mental_model, n_episodes, cost, cdf_scale, device, env_type, rng):
    # 그래디언트 추적을 끄기 위해 반드시 eval() 모드로 전환
    if env_type == "discrete":
        rollout_policy_full.q_net.eval()
    else:
        rollout_policy_full.policy.eval()
        
    expert_policy_net.eval()
    gt_mental_model.eval() 
    
    dataset = {'state': [], 'human_action': [], 'intervention': []} if env_type == "discrete" else \
              {'state': [], 'action': [], 'intervention_prob': [], 'intervention': []}
    action_size = venv.action_space.n if env_type == "discrete" else None

    # OOM 방지를 위한 필수 컨텍스트 매니저
    with torch.no_grad():
        for _ in range(n_episodes):
            obs = venv.reset() 
            done = False
            while not done:
                # 이미지/벡터에 상관없이 0.0~1.0 스케일링 및 텐서 변환 자동 수행
                obs_tensor, _ = rollout_policy_full.policy.obs_to_tensor(obs)
                obs_tensor = obs_tensor.to(device)
                
                rollout_action, _ = rollout_policy_full.predict(obs, deterministic=True)
                
                if env_type == "discrete":
                    final_probs, _ = computational_intervention_model(obs_tensor, gt_mental_model, expert_policy_net, cost, cdf_scale, env_type)
                    final_action = final_probs.argmax().item()
                    intervened = 1 if final_action != action_size else 0
                    
                    dataset['state'].append(obs[0])
                    dataset['human_action'].append(final_action)
                    dataset['intervention'].append(intervened)
                    
                    action_env = final_action if intervened else rollout_action[0]
                    action_to_take = np.array([action_env])
                else:
                    # SAC 행동 스케일 복원 (Tanh 역산) - 클리핑으로 무한대(Inf) 방지
                    rollout_action_clamped = torch.from_numpy(rollout_action).clamp(-0.99999, 0.99999)
                    rollout_action_inv = TanhBijector.inverse(rollout_action_clamped).cpu().numpy()
                    
                    final_mu, final_log_std, intervention_prob, _, _ = computational_intervention_model(obs_tensor, gt_mental_model, expert_policy_net, cost, cdf_scale, env_type)
                    intervention_prob_np = intervention_prob.squeeze(0).cpu().numpy()
                    
                    intervene = rng.choice([0, 1], p=intervention_prob_np)
                    if intervene:
                        # 인간이 개입하면 final_mu/std 기반으로 샘플링
                        action = D.Normal(final_mu, final_log_std.exp()).sample().squeeze(0).cpu().numpy()
                    else:
                        action = rollout_action_inv[0]
                        
                    dataset['state'].append(obs[0])
                    dataset['action'].append(action)
                    dataset['intervention_prob'].append(intervention_prob_np)
                    dataset['intervention'].append(intervene)
                    
                    # 시뮬레이터에 던지기 전 다시 Tanh를 씌워 [-1, 1] 범위로 복구
                    action_env = np.tanh(action) if intervene else rollout_action[0]
                    action_to_take = np.expand_dims(action_env, axis=0)

                obs, rewards, dones, infos = venv.step(action_to_take)
                done = dones[0]
                
    return {k: np.array(v) for k, v in dataset.items()}

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
    train_config = config['mile_training']
    net_arch = env_config.get('net_arch', [256, 256])
    
    print(f"[{args.env_type.upper()}] MILE 알고리즘 학습 시작 (Device: {device})")
    env = create_env(env_config['env_name'], env_config['env_params'], args.env_type, frame_stack=args.frame_stack, seed=seed)
    
    '''
    [학습 초기화 전략]
    1. policy_to_train: 로봇이 학습할 정책 (초보자 수준에서 시작)
    2. expert_policy: 가상 인간 역할 (고정된 전문가 정책)
    3. learning_mental_model: 로봇이 추측하고 학습해 나갈 인간의 마음
    4. gt_mental_model: 데이터 수집 시 가상 인간이 사용할 고정된 정답 마음 (분리 필수)
    '''
    if args.env_type == "discrete":
        policy_to_train_full = DQN.load(env_config['paths']['suboptimal_policy'], env, device=device)
        expert_policy_full = DQN.load(env_config['paths']['expert_policy'], env, device=device)
        policy_to_train_net, expert_policy_net = policy_to_train_full.q_net, expert_policy_full.q_net
        
        learning_mental_model = MentalModelPolicy(env.observation_space.shape, env.action_space.n, net_arch).to(device)
        gt_mental_model = MentalModelPolicy(env.observation_space.shape, env.action_space.n, net_arch).to(device)
        
        initial_weights = torch.load(env_config['paths']['initial_mental_model'], map_location=device)
        learning_mental_model.load_state_dict(initial_weights)
        gt_mental_model.load_state_dict(initial_weights)
        
        # 로봇 정책의 Actor와 멘탈 모델만 묶어서 학습 진행 (인간 모델은 학습 안함)
        optimizer = optim.Adam(list(policy_to_train_net.parameters()) + list(learning_mental_model.parameters()), lr=train_config['learning_rate'])
        
    else:
        policy_to_train_full = SAC.load(env_config['paths']['suboptimal_policy'], env, device=device)
        expert_policy_full = SAC.load(env_config['paths']['expert_policy'], env, device=device)
        policy_to_train_net, expert_policy_net = policy_to_train_full.policy, expert_policy_full.policy
        
        learning_mental_model = SAC.load(env_config['paths']['initial_mental_model'], device=device).policy
        gt_mental_model = SAC.load(env_config['paths']['initial_mental_model'], device=device).policy
        
        optimizer = optim.Adam(list(policy_to_train_net.actor.parameters()) + list(learning_mental_model.parameters()), lr=train_config['learning_rate'])

    cumulative_dataset = None

    for round_num in range(train_config['num_rounds']):
        print(f"\n=== Round {round_num + 1}/{train_config['num_rounds']} ===")
        
        # 1. 데이터 수집 단계 (정답 멘탈 모델 사용, eval 모드)
        additional_data = collect_synthetic_data(env, policy_to_train_full, expert_policy_net, gt_mental_model, 
                                                 train_config['episodes_per_round'], env_config['mile_params']['cost'], 
                                                 env_config['mile_params']['cdf_scale'], device, args.env_type, rng)
        
        print(f"[수집] 개입 비율: {np.mean(additional_data['intervention']) * 100:.2f}%")
        
        # 데이터 누적 (Round 기반 학습의 핵심)
        if cumulative_dataset is None:
            cumulative_dataset = additional_data
        else:
            for key in cumulative_dataset.keys():
                cumulative_dataset[key] = np.concatenate((cumulative_dataset[key], additional_data[key]))

        train_loader = DataLoader(DictDataset(cumulative_dataset), batch_size=train_config['batch_size'], shuffle=True)

        # 2. 로봇 정책 최적화 진행 단계 (train 모드로 변경하여 그래디언트 활성화)
        policy_to_train_net.train()
        learning_mental_model.train()

        for epoch in range(train_config['epochs_per_round']):
            total_loss = 0
            for batch in train_loader:
                states, _ = policy_to_train_full.policy.obs_to_tensor(batch['state'].numpy()) 
                states = states.to(device)
                ground_truth_intervention = batch['intervention'].to(device)
                optimizer.zero_grad()
                
                if args.env_type == "discrete":
                    final_probs, _ = computational_intervention_model(states, learning_mental_model, policy_to_train_net, env_config['mile_params']['cost'], env_config['mile_params']['cdf_scale'], args.env_type)
                    loss = F.nll_loss(torch.log(final_probs), batch['human_action'].to(device))
                else:
                    _, _, intervention_prob, orig_mu, orig_log_std = computational_intervention_model(states, learning_mental_model, policy_to_train_net, env_config['mile_params']['cost'], env_config['mile_params']['cdf_scale'], args.env_type)
                    # 교정된 Loss 적용: orig_mu 타겟팅
                    loss = continuous_loss_fn(intervention_prob, orig_mu, orig_log_std, batch['action'].to(device), ground_truth_intervention)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{train_config['epochs_per_round']}], Avg Loss: {total_loss / len(train_loader):.4f}", end='\r')
        print()
        
        # 라운드 종료 시 모델 개별 저장
        policy_base, policy_ext = os.path.splitext(env_config['paths']['final_trained_policy'])
        mm_base, mm_ext = os.path.splitext(env_config['paths']['final_mental_model'])
        policy_to_train_full.save(f"{policy_base}_round_{round_num + 1}{policy_ext}")
        
        if args.env_type == "discrete":
            torch.save(learning_mental_model.state_dict(), f"{mm_base}_round_{round_num + 1}{mm_ext}")
        else:
            dummy = SAC.load(env_config['paths']['initial_mental_model'], custom_objects={"policy": learning_mental_model}, device=device)
            dummy.save(f"{mm_base}_round_{round_num + 1}{mm_ext}")

if __name__ == "__main__":
    main()