from manipulator_2d import Manipulator2D
from stable_baselines import ACKTR
import os

# Gym Environment 호출
env = Manipulator2D()

# 저장된 학습 파일로부터 weight 등을 로드
model = ACKTR.load(os.path.join("ACKTR_manipulator2D_800000_getstate_minus_actionspace0_gamma05"))

# 시뮬레이션 환경을 초기화
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break

env.render()