from gymnasium.wrappers import TransformReward

class StepPenaltyRewardWrapper(TransformReward):
    def __init__(self, env, step_penalty):
        super().__init__(env, self.penalized_reward)
        self.step_penalty = step_penalty
        self.current_step = 0

    def penalized_reward(self, reward):
        return reward - self.step_penalty * self.current_step

    def step(self, action):
        self.current_step += 1
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, self.penalized_reward(reward), terminated, truncated, info

    def reset(self, **kwargs):
        self.current_step = 0
        return super().reset(**kwargs)
