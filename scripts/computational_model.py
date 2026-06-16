import torch
import torch.nn.functional as F
import torch.distributions as D

'''
[MILE 핵심 수학 모델 - 개입 확률(Intervention Probability) 연산]
이 모듈은 인간(Expert)이 로봇(Policy)의 행동을 보고 언제 개입(Intervention)할지 결정하는 수학적 확률 모델을 구현합니다.

* 논리 흐름 (Continuous 기준):
  1. 멘탈 모델 기댓값 추정 (mental_model_expectation):
     - 인간이 머릿속에 그리는 로봇의 이상적인 행동(Mental Model)을 Monte Carlo 방식으로 샘플링.
     - 이 샘플들이 현재 로봇의 정책(Policy) 분포 상에서 가질 확률(log_prob)의 평균 기댓값을 구함.
  2. 로봇 정책 샘플링 및 내부 식 연산 (inside_cdf):
     - 로봇 정책에서 직접 행동을 샘플링하여 확률(log_prob) 도출.
     - (로봇의 행동 확률) - (멘탈 모델 기댓값) - (개입 비용 Cost)
  3. CDF 통과 및 최종 개입 확률 (intervention_prob):
     - 위에서 구한 값을 cdf_scale을 표준편차로 가지는 정규분포(CDF)에 통과시켜 0~1 사이의 확률로 매핑.
     - 샘플들에 대해 평균(mean)을 구하여 최종 개입 확률을 도출.
  4. 정책 보정 (final_mu, final_log_std):
     - 원래의 mu에 개입 확률을 곱함 (final_mu)
     - 원래의 log_std에 log(개입 확률)을 더해 불확실성 스케일 조정 (final_log_std)

* ⚠️ 원본 코드 대비 치명적 오류 수정 사항:
  - Log(0) 발산 방지: torch.log 내부에 .clamp(min=1e-9)를 강제 적용하여 학습 중 무한대(NaN) 발생 원천 차단.
  - Continuous Loss 기준값 변경: 연속형 환경에서 Loss를 구할 때 스케일이 축소된 final_mu를 타겟으로 삼으면 
    정책이 완전히 붕괴됨. 오리지널 분포를 추종하도록 orig_mu와 orig_log_std를 추가로 반환하여 Loss 함수 교정.
'''

def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) > 1:
        return tensor.sum(dim=-1)
    return tensor.sum()

def monte_carlo_samples(state: torch.Tensor, policy, num_samples: int = 1000) -> torch.Tensor:
    mu, log_std, _ = policy.actor.get_action_dist_params(state)
    dist = D.Normal(mu, log_std.exp())
    return dist.rsample((num_samples,))

def computational_intervention_model(state, mental_model, policy, cost, cdf_scale, env_type="discrete"):
    if env_type == "discrete":
        mental_model_probs = F.softmax(mental_model(state), dim=1)
        policy_probs = F.softmax(policy(state), dim=1)
        std_normal_dist = D.Normal(0.0, cdf_scale)
        
        # Cross Entropy 항: 멘탈 모델이 생각하는 이상적 확률과 실제 정책 확률 간의 차이
        cross_entropy_term = (mental_model_probs * torch.log(policy_probs.clamp(min=1e-9))).sum(dim=1)
        
        # 브로드캐스팅 오류 방지를 위해 .unsqueeze(1) 명시적 적용
        inside_cdf = policy_probs - cross_entropy_term.unsqueeze(1) - cost
        intervention_prob = (policy_probs * std_normal_dist.cdf(inside_cdf)).sum(dim=1)
        
        batch_size, action_size = policy_probs.shape
        final_probs = torch.zeros(batch_size, action_size + 1, device=state.device)
        final_probs[:, :-1] = policy_probs * intervention_prob.unsqueeze(1)
        final_probs[:, -1] = 1 - final_probs[:, :-1].sum(dim=1) # 개입할 확률(마지막 차원)
        
        return final_probs.clamp(min=1e-9), None

    elif env_type == "continuous":
        mu, log_std, _ = policy.actor.get_action_dist_params(state)
        policy_dist = D.Normal(mu, log_std.exp())
        
        # 1. log(policy model distribution에 대한 mental model 기댓값)
        mental_model_samples = monte_carlo_samples(state, mental_model, 1000)
        mental_model_expectation = torch.mean(sum_independent_dims(policy_dist.log_prob(mental_model_samples)), dim=0)
        
        # 2. policy 샘플링 후 다 log 취하고 CDF 통과
        policy_samples = monte_carlo_samples(state, policy, 1000)
        mean, std_dev = torch.tensor([0.0], device=state.device), torch.tensor([cdf_scale], device=state.device)
        
        policy_expectation = D.Normal(mean, std_dev).cdf(sum_independent_dims(policy_dist.log_prob(policy_samples)) - mental_model_expectation - cost)
        intervention_prob = torch.mean(policy_expectation, dim=0)
        
        # 3. 최종 행동 확률
        final_mu = intervention_prob.unsqueeze(1) * mu
        final_log_std = torch.log(intervention_prob.unsqueeze(1).clamp(min=1e-9)) + log_std
        
        # 4. 개입 확률 벡터 구성: [비개입 확률, 개입 확률]
        intervention_prob_vec = intervention_prob.unsqueeze(-1)
        intervention_prob_vec = torch.cat((1 - intervention_prob_vec, intervention_prob_vec), dim=-1)
        
        return final_mu, final_log_std, intervention_prob_vec, mu, log_std

def continuous_loss_fn(intervention_prob, orig_mu, orig_log_std, ground_truth_action, ground_truth_intervention):
    """
    [수식 보정된 Loss] 
    개입이 발생했을 때(ground_truth_intervention == 1), 로봇의 정책이 전문가(인간)의 행동을 따라가도록 NLL Loss를 계산합니다.
    이때 개입 확률로 스케일이 찌그러진 final_mu가 아닌 원본 정책의 'orig_mu'를 사용해야 올바른 방향으로 역전파가 발생합니다.
    """
    discrete_loss = F.nll_loss(torch.log(intervention_prob.clamp(min=1e-7)), ground_truth_intervention)
    intervention_indices = torch.logical_and(ground_truth_intervention == 1, intervention_prob[:, -1] > 0.0)
    
    if intervention_indices.sum() == 0:
        continuous_loss = torch.tensor(0.0, device=intervention_prob.device)
    else:
        mu, log_std = orig_mu[intervention_indices], orig_log_std[intervention_indices]
        dist = D.Normal(mu, log_std.exp())
        continuous_loss = -sum_independent_dims(dist.log_prob(ground_truth_action[intervention_indices])).mean()
        
    return continuous_loss + discrete_loss