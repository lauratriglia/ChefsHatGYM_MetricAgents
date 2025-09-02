
from agents.agent_dqn import DQNAgent  
from rewards.attack import RewardAttack  
from rewards.defense import RewardDefense
import numpy as np
import itertools
import time
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import tensorflow as tf
import matplotlib.pyplot as plt

def dueling_lambda(a):
    return a - tf.reduce_mean(a, axis=1, keepdims=True)

class AgentDQLCustomReward(DQNAgent):
    def __init__(self, name, reward_type="attack", *args, **kwargs):
        kwargs.setdefault('state_size', 31)
        super().__init__(name, *args, **kwargs)
        
        # Select reward metric
        if reward_type == "attack":
            self.reward = RewardAttack(agent_name=self.name)
        elif reward_type == "defense":
            self.reward = RewardDefense(agent_name=self.name)
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")       
        self.current_turn_actions = []           
        self.last_action_per_player = []         
        self.last_custom_reward = 0.0  
        self.training_durations = []
        self.player_card_counts = {}     
        self.player_card_counts_next = {}
        self.match_custom_rewards = []  # Store custom rewards from each round  
        self.round_transition_indices = []  # NEW: indices of transitions in current round

    def _build_model(self, lr):
        state_input = Input(shape=(self.state_size,), name="state_input")
        x = Dense(256, activation="relu")(state_input)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        value = Dense(1, activation="linear")(x)
        advantage = Dense(self.action_size, activation="linear")(x)
        advantage_mean = Lambda(dueling_lambda, output_shape=(self.action_size,))(
            advantage
        )
        q_values = Add()([value, advantage_mean])
        model = Model(inputs=state_input, outputs=q_values)
        model.compile(loss=Huber(), optimizer=Adam(learning_rate=lr))
        return model      
    
    def request_action(self, observations):
        hand = np.array(observations["hand"]).flatten() / 13
        board = np.array(observations["board"]).flatten() / 13
        possible_actions_values = list(observations["possible_actions"])

        # Add card counts for other players
        all_players = sorted(p for p in self.player_card_counts if p != self.name)
        card_counts = [self.player_card_counts.get(p, 0) / 17 for p in all_players]
        while len(card_counts) < 3:
            card_counts.append(0)
        
        obs = np.concatenate([hand, board, card_counts])
        possible_actions_mask = np.zeros(self.action_size, dtype=np.float32)
        valid_action_indices = [
            self.all_actions.index(val) for val in possible_actions_values
        ]
        possible_actions_mask[valid_action_indices] = 1.0

        action_index = self.act(obs, possible_actions_mask, valid_action_indices)
        action_str = self.all_actions[action_index]

        # --- Reward Shaping (similar to original DQN) ---
        shaped_reward = 0.0
        if action_str.lower() == "pass":
            shaped_reward -= 1.0

        shaped_reward -= 0.02

        if (
            self.last_state is not None
            and self.last_action is not None
            and self.train
            and self.last_possible_actions is not None
        ):
            self.episode.append(
                (
                    self.last_state,
                    self.last_possible_actions,
                    self.last_action,
                    0.0,  # Placeholder, will be set in update_pizza_declared
                    obs,
                    possible_actions_mask,
                    False,
                )
            )
            self.round_transition_indices.append(len(self.episode) - 1)  # Track this transition
        self.last_state = obs
        self.last_action = action_index
        self.last_possible_actions = possible_actions_mask
        return action_index
    

    def update_player_action(self, payload):
        player = payload["player"]
        action = payload["action"]

        # Save action to be used in reward function
        self.current_turn_actions.append((player, action))

        # Track card counts for all players (lightweight tracking only)
        ob_before = payload.get("observation_before")
        ob_after = payload.get("observation_after")

        if ob_before:
            hand = ob_before.get("hand", [])
            cards_held = np.count_nonzero(hand) if isinstance(hand, np.ndarray) else sum(1 for c in hand if c != 0)
            self.player_card_counts[player] = cards_held
        if ob_after: 
            hand = ob_after.get("hand", [])
            cards_held_after = np.count_nonzero(hand) if isinstance(hand, np.ndarray) else sum(1 for c in hand if c != 0)
            self.player_card_counts_next[player] = cards_held_after

        # No need for match_memory - we'll use only the episode memory from request_action
    
    def update_pizza_declared(self, info):
        """
        Called at the end of a round. Calculate custom reward and assign it to all transitions in this round.
        """
        reward_info = {
            "player_actions": self.current_turn_actions,
            "agent_name": self.name,
        }
        if self.train:
            self.last_custom_reward = self.reward.get_reward(reward_info)
            # Assign this reward to all transitions in this round
            for idx in self.round_transition_indices:
                exp = list(self.episode[idx])
                exp[3] = self.last_custom_reward  # Set reward
                self.episode[idx] = tuple(exp)
            self.round_transition_indices = []  # Reset for next round
            self.current_turn_actions = []

    def update_match_over(self, payload):
        finishing_order = payload.get("finishing_order", [])
        scores = payload.get("scores", {})
        player_name = self.name

        self.score_history.append(payload["scores"])

        try:
            place = finishing_order.index(player_name) + 1
        except ValueError:
            place = len(finishing_order)

        self.positions.append(place)

        
        # Now do the training like the original DQN, but with per-round custom rewards
        if self.train:
            # If last_custom_reward is 1 and agent is first, set to 3
            if hasattr(self, 'last_custom_reward') and self.last_custom_reward == 3 and place == 1:
                self.last_custom_reward = 5
            start_time = time.time()
            # Mark last transition as terminal
            if (
                self.last_state is not None
                and self.last_action is not None
                and self.last_possible_actions is not None
                and len(self.episode) > 0
            ):
                self.episode.append(
                (
                    self.last_state,
                    self.last_possible_actions,
                    self.last_action,
                    self.last_custom_reward,
                    self.last_state,
                    self.last_possible_actions,
                    True,
                )
            )
            # Process all episode experiences and add to memory
            for exp in self.episode:
                self.remember(*exp)
            self.episode = []
            self.replay()
            end_time = time.time()
            elapsed = end_time - start_time
            self.training_durations.append(elapsed)
            self.match_custom_rewards.append(self.last_custom_reward)
            self.current_turn_actions = []
        self.current_turn_actions = []
        self.match_custom_rewards = []
        self.round_transition_indices = []

   