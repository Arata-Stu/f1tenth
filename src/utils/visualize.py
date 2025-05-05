import os
import yaml
import matplotlib.pyplot as plt

def visualize_speed_map(map_manager, figsize=(10,10), save_path=None):
    """YAML指定＋origin/resolution考慮でマップ上に速度可視化"""
    # YAML からマップ設定読み込み
    with open(map_manager.map_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    image_file = cfg['image']
    resolution = cfg['resolution']  # [m/pixel]
    origin_x, origin_y = cfg['origin'][0], cfg['origin'][1]

    # 画像読み込み
    img_path = os.path.join(map_manager.map_base_dir, image_file)
    img = plt.imread(img_path)
    height, width = img.shape[:2]

    # ワールド座標→ピクセル座標変換（Y軸反転補正）
    wpts = map_manager.waypoints[:, :2]
    px = (wpts[:, 0] - origin_x) / resolution
    py = height - (wpts[:, 1] - origin_y) / resolution

    # 描画
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin='upper')
    sc = ax.scatter(px, py, c=map_manager.waypoints[:, 2], cmap='bwr', s=10)

    # カラーバー
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Speed [m/s]')
    ax.set_title(f"Speed Visualization: {map_manager.map_name}")
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()
