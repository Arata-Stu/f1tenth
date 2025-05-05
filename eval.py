import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from f1tenth_gym.maps.map_manager import MapManager, MAP_DICT
from f1tenth_gym.f110_env import F110Env
from src.envs.wrapper import F110Wrapper
from src.models.TinyLidarNet import TinyLidarNet
from src.utils.helppers import convert_action



@hydra.main(config_path="config", config_name="eval", version_base="1.2")
def main(cfg: DictConfig):
    # --- 設定表示 ---
    print(OmegaConf.to_yaml(cfg))

    # --- MapManager & 環境初期化 ---
    map_cfg = cfg.envs.map
    map_manager = MapManager(
        map_name=MAP_DICT[0],
        map_ext=map_cfg.ext,
        line_type=map_cfg.line_type,
        speed=map_cfg.speed,
        downsample=map_cfg.downsample,
        use_dynamic_speed=map_cfg.use_dynamic_speed,
        a_lat_max=map_cfg.a_lat_max,
        smooth_sigma=map_cfg.smooth_sigma
    )
    env = F110Env(
        map=map_manager.map_path,
        map_ext=map_manager.map_ext,
        num_beams=cfg.num_beams,
        num_agents=1,
        params=cfg.vehicle
    )
    env = F110Wrapper(env, map_manager=map_manager)

    # --- モデルロード ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lidar_dim = cfg.num_beams
    act_dim = 2
    model = TinyLidarNet(lidar_dim, act_dim).to(device)
    ckpt = os.path.expanduser(cfg.eval.ckpt_path)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # --- 可視化設定 ---
    if cfg.visualize_lidar:
        plt.ion()
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, polar=True)
        line, = ax.plot([], [], lw=2)
        ax.set_ylim(0, cfg.max_lidar_range)

    # 変換パラメータ
    steer_range = cfg.eval.steer_range
    speed_range = cfg.eval.speed_range

    # --- 評価ループ ---
    prev_action = torch.zeros((1, act_dim), device=device)
    for map_name in MAP_DICT.values():
        print(f"\n=== マップ: {map_name} ===")
        env.update_map(map_name=map_name, map_ext=map_cfg.ext)
        obs, info = env.reset()
        terminated = truncated = False

        pbar = tqdm(desc=f"Eval {map_name}", leave=False)
        while not (terminated or truncated):
            # LiDAR 正規化
            scan_np = obs['scans'][0].astype('float32')
            scan_tensor = torch.from_numpy(scan_np).unsqueeze(0).unsqueeze(0) / 30.0
            scan_tensor = scan_tensor.to(device)

            # 推論
            with torch.no_grad():
                raw_pred = model(scan_tensor, prev_action)
            raw = raw_pred.cpu().numpy().ravel()
            steer, speed = convert_action(raw, steer_range, speed_range)
            action = np.array([[steer, speed]], dtype='float32')

            # ステップ実行
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 可視化
            if cfg.visualize_lidar:
                angles = np.linspace(-cfg.lidar_fov, cfg.lidar_fov, cfg.num_beams)
                line.set_data(angles, obs['scans'][0])
                plt.pause(0.001)
            env.render(mode=cfg.render_mode) if cfg.render_mode else env.render()

            obs = next_obs
            prev_action = torch.from_numpy(action).to(device)
            pbar.update(1)
        pbar.close()

        print(f"マップ『{map_name}』終了 terminated={terminated}, truncated={truncated}")
        input("Enterキーで次マップへ…")

if __name__ == "__main__":
    main()
