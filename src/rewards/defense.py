from rewards.reward import Reward
import numpy as np

class RewardDefense:
    rewardName = "DefenseScore"

    def __init__(self, agent_name):
        self.agent_name = agent_name

    def get_reward(self, info):
        """
        Reward is 1 if exactly two passes occur immediately before the agent's discard, otherwise -1.
        """
        try:
            actions = info.get("player_actions")  # list of (player, action)
            agent_name = info.get("agent_name")

            if not actions or not agent_name:
                return -1.0

            # Find index of agent's discard action (non-pass)
            discard_index = None
            for i, (player, action) in enumerate(actions):
                if player == agent_name and action != "pass":
                    discard_index = i
                    break

            if discard_index is None or discard_index < 2:
                return -1.0

            # Check if the two actions before discard are passes
            if actions[discard_index - 2][1] == "pass" and actions[discard_index - 1][1] == "pass":
                return 1.0
            else:
                return -1.0

        except Exception as e:
            print(f"[RewardDefense] Error: {e}")
            return -1.0