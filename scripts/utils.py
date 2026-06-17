import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import gymnasium as gym
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from imitation.data.wrappers import RolloutInfoWrapper

# 전역 디바이스 선택 함수 (어느 환경에서든 최적의 장치 할당: NVIDIA, Apple Silicon, CPU 지원)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

'''
[Box2D SWIG Float32 TypeError 해결 래퍼]
CarRacing 등 C++ Box2D를 사용하는 환경은 C언어 수준의 float32를 엄격히 요구합니다.
Numpy의 np.float32 객체를 그대로 넘기면 SWIG 바인딩에서 크래시가 발생하므로,
파이썬 내장 float()로 감싸주어 안전하게 C++ 코드로 넘겨주는 필수 래퍼입니다.
'''
class Float32ActionWrapper(gym.ActionWrapper):
    def action(self, action):
        action = np.asarray(action).reshape(-1)
        return [float(a) for a in action]

'''
[통합 환경 빌더]
1D 벡터 환경(LunarLander)과 3D 이미지 환경(CarRacing), 스택 요구 등 모든 조건을 
유연하게 처리하기 위해 VecEnv 기반으로 빌드합니다.
특히 VecTransposeImage를 적용하여 (96,96,3) 포맷의 이미지를 파이토치 CNN이 
이해할 수 있는 (3,96,96)으로 자동 변환해 줍니다.
'''
def create_env(env_name, env_params, env_type, frame_stack=1, for_imitation=False, render_mode=None, seed=42):
    def _init():
        kwargs = env_params.copy()
        if render_mode:
            kwargs['render_mode'] = render_mode

        env = gym.make(env_name, **kwargs)
        env = Monitor(env)
        
        if env_type == "continuous":
            env = Float32ActionWrapper(env)
            
        if for_imitation:
            env = RolloutInfoWrapper(env)
            
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    venv = DummyVecEnv([_init])

    # CarRacing처럼 이미지 observation을 쓰는 경우에만 적용
    if env_type == "continuous":
        venv = VecTransposeImage(venv)

    if frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)

    return venv

class DictDataset(Dataset):
    def __init__(self, data_dict):
        self.data = data_dict

    def __len__(self):
        return len(self.data['state'])

    def __getitem__(self, idx):
        item = {}
        for key in self.data:
            item[key] = self.data[key][idx]
        return item

# 이산형 환경 맞춤 멘탈 모델 (config의 net_arch에 맞춰 레이어를 동적으로 할당)
class MentalModelPolicy(nn.Module):
    def __init__(self, state_shape, action_size, net_arch=[64, 64]):
        super(MentalModelPolicy, self).__init__()
        self.state_size = np.prod(state_shape)
        layers = []
        input_dim = self.state_size
        
        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        self.fc_out = nn.Linear(input_dim, action_size)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.feature_extractor(x)
        return self.fc_out(x)