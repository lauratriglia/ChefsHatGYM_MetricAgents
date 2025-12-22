class RewardVitality:
    rewardName = "VitalityScore"

    def __init__(self, agent_name):
        self.agent_name = agent_name

    def get_reward(self, info):
        """
        Computes reward based on vitality (number of discards by the agent).
        Returns reward, attack, defense, vitality.
        Reward: 3 if vitality = 1, else -0.05
        """
        try:
            actions = info.get("player_actions")  # list of (player, action)
            agent_name = info.get("agent_name")

            if not actions or not agent_name:
                return -0.05, 0, 0, 0

            # Count vitality: number of discards by the agent (non-pass)
            discard_index = None
            vitality = 0
            for i, (player, action) in enumerate(actions):
                if player == agent_name and action != "pass":
                    discard_index = i
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
            
            vitality = sum(1 for player, action in actions if player == agent_name and action != "pass")
            # Reward logic
            if vitality == 1:
                reward = 3.0
            else:
                reward = -0.05

            return reward, attack, defense, vitality

        except Exception as e:
            print(f"[RewardVitality] Error: {e}")
            return -0.05, 0, 0, 0
