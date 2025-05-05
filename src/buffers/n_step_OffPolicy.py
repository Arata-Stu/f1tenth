from collections import deque
import numpy as np
from .OffPolicy import OffPolicyBuffer

class NStepOffPolicyBuffer(OffPolicyBuffer):
    def __init__(self, size, state_dim, action_dim, device, n_step, gamma):
        super().__init__(size, state_dim, action_dim, device)
        self.n_step = n_step
        self.gamma = gamma
        # 一時的に最大 n ステップ分をためておく deque
        self.n_buffer = deque(maxlen=n_step)

    def add(self, state, action, reward, next_state, done, log_prob=None):
        # まず deque に追加
        self.n_buffer.append((state, action, reward, next_state, done, log_prob))
        # デックが n ステップ分たまったら古いものをまとめて保存
        if len(self.n_buffer) == self.n_step:
            # 0ステップ目の state, action
            s0, a0, *_ = self.n_buffer[0]
            # nステップ目の next_state, done
            *_, sn, done_n, lp_n = self.n_buffer[-1]

            # nステップ目の next_state が done なら次の state は 0 ベクトル
            next_state = np.zeros_like(sn) if done_n else sn


            # n-step リターンを計算
            discounted_reward = 0.0
            for idx, (_, _, r, _, _, _) in enumerate(self.n_buffer):
                discounted_reward += (self.gamma**idx) * r

            # オフポリ用バッファに入れる
            # （log_prob は次ステップで actor から再計算するので None でも可）
            super().add(
                state=s0,
                action=a0,
                reward=np.array([[discounted_reward]], dtype=np.float32),
                next_state=sn,
                done=np.array([[done_n]], dtype=np.float32),
                log_prob=None
            )
        # エピソード終端なら残りもフラッシュ
        if done:
            while len(self.n_buffer) > 0:
                # 同様に長さ < n のリターンを計算して保存
                L = len(self.n_buffer)
                s0, a0, *_ = self.n_buffer[0]
                *_, sn, done_n, lp_n = self.n_buffer[-1]
                discounted_reward = 0.0
                for idx, (_, _, r, _, _, _) in enumerate(self.n_buffer):
                    discounted_reward += (self.gamma**idx) * r
                super().add(
                    state=s0,
                    action=a0,
                    reward=np.array([[discounted_reward]], dtype=np.float32),
                    next_state=sn,
                    done=np.array([[done_n]], dtype=np.float32),
                    log_prob=None
                )
                self.n_buffer.popleft()
