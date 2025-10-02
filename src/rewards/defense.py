from rewards.reward import Reward
import numpy as np

class RewardDefense:
    rewardName = "DefenseScore"

    def __init__(self, agent_name):
        self.agent_name = agent_name

    def get_reward(self, info):
        """
        Reward is 1 if exactly two passes occur immediately before the agent's discard, otherwise -1.
        Also returns attack (passes after discard), defense (passes before discard),
        and vitality (number of discards by the agent).
        """
        try:
            actions = info.get("player_actions")  # list of (player, action)
            agent_name = info.get("agent_name")

            if not actions or not agent_name:
                return -1.0, 0, 0, 0

            # Find index of agent's discard action (non-pass)
            discard_index = None
            for i, (player, action) in enumerate(actions):
                if player == agent_name and action != "pass":
                    discard_index = i
                    break

            if discard_index is None or discard_index < 2:
                return -1.0, 0, 0, 0

            # Defense: passes before discard
            defense = 0
            for j in range(discard_index):
                if actions[j][1] == "pass":
                    defense += 1

            # Attack: passes after discard
            attack = 0
            for j in range(discard_index + 1, len(actions)):
                if actions[j][1] == "pass":
                    attack += 1

            # Vitality: number of discards by the agent
            vitality = sum(1 for player, action in actions if player == agent_name and action != "pass")

            # Check if the two actions before discard are passes
            if actions[discard_index - 2][1] == "pass" and actions[discard_index - 1][1] == "pass":
                reward = 3.0
            else:
                reward = -0.05

            return reward, attack, defense, vitality

        except Exception as e:
            print(f"[RewardDefense] Error: {e}")
            return -1.0, 0, 0, 0