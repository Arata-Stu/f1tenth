import os
import numpy as np
import h5py
import hdf5plugin
import hydra
from omegaconf import DictConfig, OmegaConf
from src.envs.envs import make_env
from f1tenth_gym.maps.map_manager import MapManager
from src.planner.purePusuit import PurePursuitPlanner

@hydra.main(config_path="config", config_name="collect_data", version_base="1.2")
def main(cfg: DictConfig):
    # 設定表示
    print(OmegaConf.to_yaml(cfg))

    # --- 環境とプランナーの初期化 ---
    map_cfg = cfg.envs.map
    map_manager = MapManager(
        map_name=map_cfg.name,
        map_ext=map_cfg.ext,
        speed=map_cfg.speed,
        downsample=map_cfg.downsample,
        use_dynamic_speed=map_cfg.use_dynamic_speed,
        a_lat_max=map_cfg.a_lat_max,
        smooth_sigma=map_cfg.smooth_sigma
    )
    env = make_env(env_cfg=cfg.envs, map_manager=map_manager, param=cfg.vehicle)

    wheelbase = cfg.planner.wheelbase if hasattr(cfg.planner, 'wheelbase') else (0.17145 + 0.15875)
    lookahead = cfg.planner.lookahead if hasattr(cfg.planner, 'lookahead') else 0.3
    planner = PurePursuitPlanner(
        wheelbase=wheelbase,
        map_manager=map_manager,
        lookahead=lookahead,
        max_reacquire=cfg.planner.max_reacquire if hasattr(cfg.planner, 'max_reacquire') else 20.0
    )

    # レンダリング有無の設定（configから取得、未指定時はTrue）
    render_flag = cfg.render 

    # 初期観測のリセット
    obs = env.reset()

    # --- HDF5 ファイルの準備 ---
    out_dir = cfg.output_dir 
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{map_cfg.name}_speed{map_cfg.speed}_look{lookahead}.h5"
    out_path = os.path.join(out_dir, filename)

    f = h5py.File(out_path, 'w')
    scans_dset = f.create_dataset(
        name='scans',
        shape=(0, 1080),
        maxshape=(None, 1080),
        dtype='float32',
        chunks=(1, 1080),
        **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    )
    actions_dset = f.create_dataset(
        name='actions',
        shape=(0, 2),
        maxshape=(None, 2),
        dtype='float32',
        chunks=(1, 2),
        **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    )

    # --- データ収集ループ ---
    idx = 0
    while True:
        # プランナーで制御量を取得
        steer, speed = planner.plan(obs, gain=0.20)

        # LiDAR スキャンとアクションを整形
        scan = obs['scans'][0].astype('float32').reshape(1, -1)
        action = np.array([steer, speed], dtype='float32').reshape(1, -1)

        # データセットを拡張して書き込み
        scans_dset.resize(idx + 1, axis=0)
        actions_dset.resize(idx + 1, axis=0)
        scans_dset[idx] = scan
        actions_dset[idx] = action

        # 次ステップ実行
        next_obs, reward, terminated, truncated, info = env.step(action.flatten())
        if terminated or truncated:
            print(f"terminated after {idx+1} steps, saved to {out_path}")
            break

        obs = next_obs
        idx += 1
        # レンダリング
        if render_flag:
            env.render(mode=cfg.render_mode)

    # --- ファイルをクローズ ---
    f.close()

if __name__ == '__main__':
    main()
