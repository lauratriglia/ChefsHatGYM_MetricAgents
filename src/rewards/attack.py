import sys
sys.path.insert(0, '/usr/local/src/robot/cognitiveInteraction/ChefHatGYM_v3/ChefsHatGYM/src')

from rewards.reward import Reward
import numpy as np

class RewardAttack:
    rewardName = "AttackScore"

    def __init__(self, agent_name):
        self.agent_name = agent_name

    def get_reward(self, info):
        """
        Reward is 1 if 3 passes occur after the agent's discard, otherwise -0.05.
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

            if discard_index is None:
                return -3.0

            # Count passes after discard
            passes_after_discard = 0
            for _, (player, action) in enumerate(actions[discard_index + 1:]):
                if action == "pass":
                    passes_after_discard += 1
                else:
                    break
                if passes_after_discard == 3:
                    break

            if passes_after_discard == 3:
                return 3.0
            else:
                return -2.0

        except Exception as e:
            print(f"[RewardAttack] Error: {e}")
            return -0.05
        

    
    def get_reward_weak(self, info):
        """
        Calculate the attack reward based on how many passes happen
        after the agent's discard in the action sequence.

        :param actions_list: List of tuples [(player, action), ...] in order
        :param agent_name: The name of the agent to compute reward for
        :return: float reward normalized between 0 and 1
        """
        try:
            actions = info.get("player_actions")  # list of (player, action)
            agent_name = info.get("agent_name")

            if not actions or not agent_name:
                return 0.0

            # Find index of agent's discard action (non-pass)
            discard_index = None
            for i, (player, action) in enumerate(actions):
                if player == agent_name and action != "pass":
                    discard_index = i
                    break

            if discard_index is None:
                # Agent did not discard this round
                return -2.0

            # Count passes after discard (max 3 passes)
            passes_after_discard = 0
            for _, (player, action) in enumerate(actions[discard_index + 1:]):
                if action == "pass":
                    passes_after_discard += 1
                else:
                    # Non-pass action after discard ends counting
                    break
                if passes_after_discard == 3:
                    break

            return passes_after_discard / 3.0  # normalized reward

        except Exception as e:
            print(f"[RewardAttack] Error: {e}")
            return 0.0
