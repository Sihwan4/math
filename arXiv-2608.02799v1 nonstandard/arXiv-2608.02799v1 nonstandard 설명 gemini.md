# A Hyperfinite Framework for Score-Based Generative Modeling AUTHOR: Sunder Ram Krishnan (Kerala School of Mathematics) --- ### **Overview of

_Gemini · 2 messages · exported 2026년 11월 5일_

### You

- arXiv-2608.02799v1 nonstandard

### Gemini

이 논문은 현대 생성 모델의 핵심인 스코어 기반 확산 모델(Score-based Diffusion Models)을 비표준 해석학(Nonstandard Analysis, NSA)의 **초유한 격자(Hyperfinite Grid)** 위에서 재구성한 논문이야.

기존 확산 모델 이론(Song et al., Anderson 역시간 SDE 등)은 연속 시간 SDE, 측도론적 확률 미적분학, 복잡한 극한 과정에 의존했어. 이 논문은 무한소(infinitesimal) 시간 간격 $\delta t$와 공간 간격 $\delta x$를 가진 초유한 격자 위에서 이산적 대수학(discrete algebra)으로 모델을 전개한 뒤, 표준 부분 사상($\operatorname{st}$ 또는 ${}^\circ$)을 취해 연속 이론과 완벽히 대응함을 증명해.

논문의 핵심 흐름과 주요 증명 결과를 직관적이면서도 엄밀하게 정리해 줄게.

## 1. 핵심 철학과 수학적 세팅

- **화이트박스(White-Box) 관점**: 연속 시간 브라운 운동과 확률 미분 방정식(SDE)을 추상적 극한으로 보지 않고, 초유한 보폭을 갖는 내적 랜덤 워크(internal random walk)의 '그림자(shadow, 표준 부분)'로 해석해.
- **시간/공간 격자 설정**:
  - 시간 간격: 무한소 $\delta t = 1/N$ ($N \in {}^*\mathbb{N} \setminus \mathbb{N}$).
  - 공간 격자: $\mathbb{G} \subset {}^*\mathbb{R}^d$, 격자 간격 $\delta x = C\sqrt{\delta t}$ (포물형 스케일링, CFL 조건 만족).
- **격자 미분 연산자**: 전진 차분 $\Delta_i^+ u(x) = \frac{u(x + \delta x \mathbf{e}_i) - u(x)}{\delta x}$ 등을 사용해 미분 연산자를 격자 상의 유한 차분으로 다뤄.

## 2. 논문의 주요 이론적 성과 5가지

### ① 무한소 생성원과 포커-플랑크(Fokker–Planck) 쌍대성

격자 동역학 $X_{t+\delta t} = X_t + \Delta X_t$에서 점프 모멘트 조건을 이용해 테스트 함수 $\varphi \in C_c^3$의 전진 기댓값을 계산해:

$$
\mathbb{E}_{\mathrm{int}}[{}^*\varphi(X_{t+\delta t}) - {}*\varphi(x) \mid X_t = x] = \delta t \,\,{}^*\mathcal{L}_{t} {}^*\varphi(x) + O(\delta t^{3/2})
$$

여기서 격자 생성원 ${}^{\ast}\mathcal{L}_t$ 는 전진 차분 연산자 $\Delta_i^+$, $\Delta_i^+\Delta_j^+$ 로 표현돼. 이후 격자 상에서의 이산 부분적분(Summation by parts / Adjoint identity)을 사용해 확률 보존 법칙을 취하면, 복잡한 측도론적 극한 없이 대수적 전개만으로 표준 연속 시간의 포커-플랑크 방정식(Fokker–Planck Equation) $\frac{\partial p}{\partial t} = \mathcal{L}_t^{\ast}p $가 그대로 유도돼.

### ② 역시간 평균 항등식(Backward Mean Identity)과 역시간 SDE 유도

확산 모델의 핵심인 "시간 역전(time-reversal)"을 초유한 격자에서 경로별(pathwise)로 계산해. 도착점 $X_{t+\delta t} = y$가 주어졌을 때 뒤로 가는 증분 $h = X_t - X_{t+\delta t}$의 조건부 기댓값을 계산하면:

$$
\mathbb{E}_{\text{int}}[X_t^i - X_{t+\delta t}^i \mid X_{t+\delta t} = y] = \beta_i(t, y) \delta t + O(\delta t^{3/2})
$$

여기서 역시간 표류 계수(backward drift) $\beta$는 다음과 같이 명시적으로 도출돼:

$$
\beta_i(t,y) = -{}^*b_i(t,y) + \sum_{j=1}^d {}^*a_{ij}(t,y) \Delta^+_j \log {}^*p_t(y) + \sum_{j=1}^d \Delta^+_j {}^*a_{ij}(t,y)
$$

논문에서는 이를 두 가지 방식으로 증명해:

1. **야코비안 섭동(Jacobian perturbation) 분석**: 미시적 점프 $h$를 테일러 전개하고 베이즈 정리와 국소 경로 밀도 변화(야코비안 행렬식의 역수)를 곱해 직접 계산.
2. **격자 플럭스(Flux density) 분석**: 격자 확률 유속 $\Phi_v(x) = p_t(x)p(x, x+v)$에 이산 테일러 전개를 적용.

이 결과로 나오는 스코어 항 $\Delta_j^+ \log {}^*p_t$가 바로 역과정에서 엔트로피를 되돌리기 위해 필요한 보정값이야. 이를 표준 부분으로 넘기면 앤더슨(Anderson, 1982)의 역시간 SDE가 자연스럽게 성립해.

### ③ 내적 스코어 매칭(Internal Score Matching)과 생성적 수렴

실제 생성 모델에서는 알지 못하는 스코어 $\Delta^+ \log p_t$를 신경망 $s_\theta$로 학습해(Denoising Score Matching, DSM).

- **직교 투영(Orthogonal Projection)**: 초유한 힐베르트 공간 $L^2$에서 DSM 손실 함수 $\mathcal{I}(\theta)$를 최소화하는 것은, 노이즈가 낀 조건부 스코어 $\Delta^+ \log P_{t\vert{}0}$를 초기값 $x_0$에 독립적인 공간으로 직교 투영하는 것과 완벽히 동치임을 증명해.
- 따라서 손실 함수 $\mathcal{I}(\theta) \approx 0$이면, 측정 가능한 거의 모든 $(x, t)$에서 $s_\theta(x, t) \approx \Delta^+ \log p_t(x)$가 성립해.
- 이를 역시간 격자 재귀식에 대입하면, 표준 부분 과정 $\bar{Y}*1 = \lim*{\epsilon \to 0} \bar{Y}_{1-\epsilon}$의 분포가 원본 데이터 분포 ${}^\circ q$로 정확히 수렴(Generative Convergence)함을 보여줘.

### ④ 초유한 기르사노프(Girsanov) 정리와 우도(Likelihood) 연결

연속 확률론에서 기르사노프 정리는 마틴게일 적분과 노비코프(Novikov) 조건 같은 까다로운 적분 가능성 검증이 필요해. 하지만 초유한 격자에서는 전이 확률의 국소 지수 틸팅(exponential tilt) 비율의 곱으로 라돈-니코딤 도함수를 직접 계산할 수 있어:

- 국소 상대 엔트로피(KL divergence)를 최소화하면서 원하는 표류 계수를 유도하는 유일한 전이 확률이 지수 틸트 형태임을 증명해.
- 이를 경로 전체로 확장하면 경로 로그 우도비 $\log \frac{d\mathbb{Q}}{d\mathbb{P}}$가 마틴게일 증분 항과 2차 보정 항으로 정확히 분해돼.
- 기대값을 취하면 마틴게일 항은 타워 법칙(tower property)으로 사라지고, 결과적으로 **피셔 스코어 매칭 오차(Fisher divergence)를 최소화하는 것이 경로 대리 우도(Surrogate Likelihood)의 하한을 최대화하는 것**과 정확히 연결됨을 보여줘.

### ⑤ 2차 일관성(Second-Order Consistency)과 첨도(Kurtosis, $\kappa=3$)의 역할

논문의 매우 흥미로운 분석 중 하나야. 공간 1차원에서 오일러 증분 $h = b\,\delta t + \sigma\sqrt{\delta t}\,\xi$에 대해, 마스터 방정식 $P(t+\delta t, x) = \mathbb{E}_\xi[P(t, x-h)]$을 반군(semigroup) 2차 전개식과 비교했을 때 오차를 계산했어:

$$
\operatorname{st}\left( \frac{P(t+\delta t,x) - \left( P + \delta t \mathbb{A} P + \frac{\delta t^2}{2} \mathbb{A}^2 P \right)}{\delta t^2} \right) = \frac{\kappa - 3}{24} a^2 \frac{\partial^4 p}{\partial x^4}
$$

- **1차 일관성**: 노이즈 증분의 1차·2차 모멘트($\mathbb{E}[\xi]=0, \mathbb{E}[\xi^2]=1$)만 맞추면 충분해.
- **2차 일관성**: 4차 모멘트(첨도)인 $\kappa = \mathbb{E}[\xi^4]$가 반드시 3(가우시안 값)이어야만 $\delta t^2$ 오차 항의 주성분이 소거돼.
- 라데마허(Rademacher, $\pm 1$) 노이즈는 $\kappa=1$이므로 2차 수렴에서 분산 오차(dispersion error)가 남아. 반면 특정 이산 3점 분포($\pm\sqrt{3}$ 확률 1/6, $0$ 확률 2/3)는 $\kappa=3$을 만족해 가우시안처럼 2차 오차를 제거할 수 있음을 짚어내.

## 3. 총평

이 논문은 비표준 해석학을 단순한 "근사 도구"로 쓰는 게 아니라, **초유한 이산 격자 자체를 본체로 두고 연속 이론을 그 그림자(standard part)로 규명**하는 방식으로 확산 모델의 복잡한 기초 이론(역시간 SDE 유도, 기르사노프 정리, DSM과 MLE의 동치 관계)을 유한 차원 대수 계산 수준으로 명쾌하게 풀어낸 연구야.

특히 샘플링 오차 분석에서 첨도 $\kappa=3$ 조건을 도출해 낸 부분은 향후 새로운 고계도(higher-order) 이산 샘플링 알고리즘 설계에도 실질적인 통찰을 주는 대목이야.
