from f1tenth_gym.maps.map_manager import MapManager #/f1tenth_gym/map_manager.py から import
from f1tenth_gym.f110_env import F110Env #/f1tenth_gym/f110_env.py から import
from src.envs.wrapper import F110Wrapper #/src/envs/wrapper.py から import

map_name = "BrandsHatch"  # マップ名
map_ext = ".png"  # 拡張子
num_beams = 1080 ## 2d Lidarのビーム数
speed = 8.0 # m/s
num_agents = 1 ## エージェントの数 1: 自車両
downsample = 1 # way_point を 1/downsample にする 1: そのまま, 2: 1/2 にする

## 車両のパラメータ (公式から引用)
vehicle_params = {
    "mu": 1.0489,
    "C_Sf": 4.718,
    "C_Sr": 5.4562,
    "lf": 0.15875,
    "lr": 0.17145,
    "h": 0.074,
    "m": 3.74,
    "I": 0.04712,
    "s_min": -1.0,
    "s_max": 1.0,
    "sv_min": -3.2,
    "sv_max": 3.2,
    "v_switch": 7.319,
    "a_max": 9.51,
    "v_min": -5.0,
    "v_max": 10.0,
    "width": 0.31,
    "length": 0.58
}

## マップを管理するクラス
map_manager = MapManager(map_name=map_name, map_ext=map_ext, speed=speed, downsample=downsample) 
## F1tenthのgym環境
env = F110Env(map=map_manager.map_path, map_ext=map_manager.map_ext, num_beams=num_beams, num_agents=num_agents, params=vehicle_params)
env = F110Wrapper(env, map_manager=map_manager)
obs, info = env.reset()

from src.planner.purePusuit import PurePursuitPlanner #/src/planner/pure_pursuit.py から import

wheelbase=(0.17145+0.15875)
planner = PurePursuitPlanner(wheelbase=wheelbase, map_manager=map_manager, lookahead=0.3 ,max_reacquire=20.) 

import matplotlib.pyplot as plt
import numpy as np

# 可視化用の設定
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
line, = ax.plot([], [], lw=2)
ax.set_ylim(0, 30)  # LiDARの最大距離（適宜調整）

# メインループ
while True:
    actions = []
    steer, speed = planner.plan(obs, gain=0.20)
    action = [steer, speed]
    actions.append(action)
    next_obs, reward, terminated, truncated, info = env.step(np.array(actions))

    # LiDARデータの可視化
       # LiDARデータの可視化
    scan = obs["scans"][0]  # (1080,)
    angles = np.linspace(-2.35, 2.35, len(scan))  # LiDARの実際の視野角に合わせて修正
    line.set_data(angles, scan)
    plt.pause(0.001)


    if terminated or truncated:
        print("terminated")
        break

    obs = next_obs
    # env.render()
