# F1tenth-RL

プロジェクトの概要を一文で説明します。

## 目次
- [プロジェクト概要](#プロジェクト概要)
- [前提条件](#前提条件)
- [環境セットアップ](#環境セットアップ)
- [実行方法](#実行方法)


## プロジェクト概要

F1tenth gym環境を用いた強化学習を行うプロジェクトです。

## 前提条件

- venv Python 3.11 
- Ubuntu 22.04

※ 上記の環境のみで検証しています。他の環境でも問題ないと考えられます。

## 環境セットアップ

### 1. リポジトリのクローン
```bash
git clone https://github.com/Arata-Stu/F1tenth-RL.git
cd F1tenth-RL
```

### 2. 仮想環境の作成と有効化（Python）
```bash
## Python3.11 のインストール
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt update
# sudo apt install python3.11 python3.11-venv python3.11-dev

python3.11 -m venv env
source env/bin/activate  
```

### 3. 依存ライブラリのインストール
```bash
pip install -r requirements.txt
```


## 実行方法

`tutorial/` ディレクトリに、Gym の基本的な動作確認用コードや強化学習アルゴリズムの実装が含まれています。まずは以下のスクリプトを実行することをおすすめします。


### 学習
'config/train.yaml'を確認すると学習設定についてわかります。Hydraというパケージを用いて動的にパラメータを変更できるように変えています。
```bash
## 例　agent: SAC, reward: progress buffer: OffPolicy Buffer
## map: Austin
python3 train.py \
agent=sac \
reward=progress \
buffer=off_policy \
envs.map.name=Austin
```
