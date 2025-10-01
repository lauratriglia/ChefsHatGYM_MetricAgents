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
        Computes reward based on passes after the agent's discard.
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
            vitality = 0
            for i, (player, action) in enumerate(actions):
                if player == agent_name and action != "pass":
                    discard_index = i
                    vitality += 1
                    break

            if discard_index is None:
                return -1.0, 0, 0, vitality  # No discard

            # Defense: count passes before discard
            defense = sum(1 for _, action in actions[:discard_index] if action == "pass")

            # Attack: count passes after discard
            attack = 0
            for _, action in actions[discard_index + 1:]:
                if action == "pass":
                    attack += 1
                else:
                    break
                if attack == 3:
                    break

            # Reward logic
            if attack == 3:
                reward = 3.0
            else:
                reward = -0.01

            return reward, attack, defense, vitality

        except Exception as e:
            print(f"[RewardAttack] Error: {e}")
            return -0.05, 0, 0, 0
        

    
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
