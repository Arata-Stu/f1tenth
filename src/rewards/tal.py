from .base import RewardBase
from ..planner.purePusuit import PurePursuitPlanner

class TALReward(RewardBase):
    def __init__(self, map_manager, steer_range, speed_range, steer_w=0.4, speed_w=0.4, bias= 0.25, ratio=1.0):
        super().__init__()
        self.map_manager = map_manager
        self.steer_range = steer_range
        self.speed_range = speed_range
        self.steer_w = steer_w
        self.speed_w = speed_w
        self.bias = bias
        self.ratio = ratio

        wheelbase=(0.17145+0.15875)
        self.planner = PurePursuitPlanner(wheelbase=wheelbase, map_manager=map_manager, lookahead=0.6 ,max_reacquire=20.) 

    def get_reward(self, pre_obs, obs, action):
        base_reward = super().get_reward(obs, pre_obs)

        pp_action = self.planner.plan(pre_obs, id=0, gain=0.2)
        
        
        steer_reward =  (abs(pp_action[0] - action[0]) / self.steer_range)  * self.steer_w
        throttle_reward =   (abs(pp_action[1] - action[1]) / self.speed_range) * self.speed_w

        reward = self.bias - steer_reward - throttle_reward
        reward = max(reward, 0) # limit at 0

        reward *= self.ratio
        return base_reward + reward