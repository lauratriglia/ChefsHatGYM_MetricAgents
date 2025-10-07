class RewardVitality:
    rewardName = "VitalityScore"

    def __init__(self, agent_name):
        self.agent_name = agent_name

    def get_reward(self, info):
        """
        Computes reward based on vitality (number of discards by the agent).
        Returns reward, attack, defense, vitality.
        Reward: 3 if vitality > 1, else -0.05
        """
        try:
            actions = info.get("player_actions")  # list of (player, action)
            agent_name = info.get("agent_name")

            if not actions or not agent_name:
                return -0.05, 0, 0, 0

            # Count vitality: number of discards by the agent (non-pass)
            vitality = sum(1 for player, action in actions if player == agent_name and action != "pass")

            # Dummy attack/defense for compatibility
            attack = 0
            defense = 0

            # Reward logic
            if vitality == 1:
                reward = 3.0
            else:
                reward = -0.05

            return reward, attack, defense, vitality

        except Exception as e:
            print(f"[RewardVitality] Error: {e}")
            return -0.05, 0, 0, 0
