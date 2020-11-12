import os

import numpy as np
from stable_baselines import A2C, PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.callbacks import EvalCallback
from manipulator_2d import Manipulator2D

from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

import matplotlib.pyplot as plt

# Gym Environment 호출
env = Manipulator2D()

# Gym env의 action_space로부터 action의 개수를 알아낸다.
n_actions = env.action_space.shape[-1]
param_noise = None

best_mean_reward, n_steps = -np.inf, 0

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(env, log_dir)

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./acktr_cartpole_tensorboard_gamma05/")

time_steps = 800000

eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=500,
                             deterministic=True, render=False)
model.learn(total_timesteps=time_steps, callback=eval_callback)

# 학습된 결과를 저장한다.
# 연습 : 400,000 timestep 동안 학습한 결과가 아닌, 학습 도중 가장 좋은 reward 값을 반환한 policy network를 저장하려면 어떻게 해야할까요?
# Tip : learn 함수에 callback function을 사용해봅시다.
model.save("ACKTR_manipulator2D_800000_getstate_minus_actionspace0_gamma05__")

#results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "A2C LunarLander")
#plt.show()
