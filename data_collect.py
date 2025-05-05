import os
import numpy as np
import h5py
import hdf5plugin
import hydra
from omegaconf import DictConfig, OmegaConf
from src.envs.envs import make_env
from f1tenth_gym.maps.map_manager import MapManager
from src.planner.purePursuit import PurePursuitPlanner

# マップ辞書
MAP_DICT = {
    0: 'Austin',        1: 'BrandsHatch', 2: 'Budapest',
    3: 'Catalunya',     4: 'Hockenheim',  5: 'IMS',
    6: 'Melbourne',     7: 'MexicoCity',  8: 'Monza',
    9: 'MoscowRaceway',10: 'Nuerburgring',11: 'Oschersleben',
    12: 'Sakhir',       13: 'SaoPaulo',   14: 'Sepang',
    15: 'Silverstone',  16: 'Sochi',      17: 'Spa',
    18: 'Spielberg',    19: 'YasMarina',  20: 'Zandvoort'
}

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

    wheelbase = getattr(cfg.planner, 'wheelbase', 0.17145 + 0.15875)
    lookahead = getattr(cfg.planner, 'lookahead', 0.3)
    planner = PurePursuitPlanner(
        wheelbase=wheelbase,
        map_manager=map_manager,
        lookahead=lookahead,
        max_reacquire=getattr(cfg.planner, 'max_reacquire', 20.0)
    )

    # レンダリング設定
    render_flag = cfg.render
    render_mode = cfg.render_mode

    # 出力ディレクトリ
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    num_episodes = cfg.num_episodes
    for ep in range(num_episodes):
        # マップ更新
        map_id = ep % len(MAP_DICT)
        name = MAP_DICT[map_id]
        env.update_map(map_name=name, map_ext=map_cfg.ext)

        # 初期観測と前回アクション
        obs = env.reset()
        prev_action = np.zeros((1, 2), dtype='float32')
        idx = 0
        truncated = False

        # ファイル名にエピソード番号を付与
        filename = f"{name}_speed{map_cfg.speed}_look{lookahead}_ep{ep}.h5"
        out_path = os.path.join(out_dir, filename)
        f = h5py.File(out_path, 'w')
        scans_dset = f.create_dataset('scans', shape=(0,1080), maxshape=(None,1080), dtype='float32', chunks=(1,1080), **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
        prev_dset = f.create_dataset('prev_actions', shape=(0,2), maxshape=(None,2), dtype='float32', chunks=(1,2), **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
        actions_dset = f.create_dataset('actions', shape=(0,2), maxshape=(None,2), dtype='float32', chunks=(1,2), **hdf5plugin.Blosc(cname='zstd', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))

        # データ収集ループ
        while True:
            steer, speed = planner.plan(obs, gain=0.20)
            action = np.array([steer, speed], dtype='float32').reshape(1,-1)
            scan = obs['scans'][0].astype('float32').reshape(1,-1)

            # 保存
            scans_dset.resize(idx+1, axis=0)
            prev_dset.resize(idx+1, axis=0)
            actions_dset.resize(idx+1, axis=0)
            scans_dset[idx] = scan
            prev_dset[idx] = prev_action
            actions_dset[idx] = action

            next_obs, reward, terminated, truncated, info = env.step(action.flatten())
            if truncated:
                print(f"Episode {ep} truncated (failure), discarding data: {out_path}")
            if terminated or truncated:
                # 終了
                f.close()
                # 失敗時はファイル削除
                if truncated:
                    os.remove(out_path)
                break

            # 準備
            obs = next_obs
            prev_action = action
            idx += 1
            if render_flag:
                env.render(mode=render_mode) if render_mode else env.render()

        # エピソード完了ログ
        if not truncated:
            print(f"Episode {ep} completed successfully, saved to {out_path}")

if __name__ == '__main__':
    main()
