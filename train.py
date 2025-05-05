import os
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.h5_dataset import H5Dataset
from src.models.TinyLidarNet import TinyLidarNet

@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # HDF5 ファイル自動探索
    data_dir = cfg.data.data_dir
    files = glob.glob(os.path.join(data_dir, "*", "*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files under {data_dir}")
    print(f"Found {len(files)} H5 files.")

    # データローダ
    dataset = H5Dataset(files)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデル初期化
    lidar_dim = dataset[0]['scan'].shape[0]
    act_dim = dataset[0]['prev'].shape[0]
    model = TinyLidarNet(lidar_dim, act_dim).to(device)

    # 最適化・損失
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.MSELoss()

    # ベストモデル保存先
    best_loss = float('inf')
    ckpt_path = cfg.train.ckpt_path
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # トレーニングループ
    for epoch in range(cfg.train.epochs):
        epoch_loss = 0.0
        # バッチ進捗表示
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}", leave=False):
            model.train()
            scan = batch['scan'].to(device).unsqueeze(1)  # (B,1,L)
            prev = batch['prev'].to(device)               # (B,A)
            target = batch['action'].to(device)           # (B,A)

            pred = model(scan, prev)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * scan.size(0)

        avg_loss = epoch_loss / len(dataset)
        tqdm.write(f"Epoch {epoch+1}/{cfg.train.epochs} loss={avg_loss:.6f}")

        # ベストモデルの保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)
            tqdm.write(f"Best model updated (loss={best_loss:.6f}) -> {ckpt_path}")

    print(f"Training completed. Best model saved at {ckpt_path} (loss={best_loss:.6f})")

if __name__ == "__main__":
    main()
