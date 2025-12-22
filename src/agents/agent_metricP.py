from agents.agent_dqn import DQNAgent  
from rewards.attack import RewardAttack  
from rewards.defense import RewardDefense
from rewards.vitality import RewardVitality
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

class AgentDQLMetrics(DQNAgent):
    def __init__(self, name, reward_type="attack", *args, **kwargs):
        # New state vector: hand(13) + board(13) + card_counts(3) + deck_counts(12) = 41
        kwargs['state_size'] = 43 # Force state size to 43 for consistency with deck counts
        super().__init__(name, *args, **kwargs)
        
        # Select reward metric
        if reward_type == "attack":
            self.reward = RewardAttack(agent_name=self.name)
        elif reward_type == "defense":
            self.reward = RewardDefense(agent_name=self.name)
        elif reward_type == "vitality":
            self.reward = RewardVitality(agent_name=self.name)
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")       
        self.current_turn_actions = []           
        self.last_action_per_player = []         
        self.last_custom_reward = 0.0  
        self.training_durations = []
        self.player_card_counts = {}     
        self.player_card_counts_next = {}
        self.rewards = []  # Store rewards per match
        self.match_custom_rewards = []  # Store custom rewards from each round
        self.attack = []
        self.defense = []
        self.vitality = []
        self.round_transition_indices = []  # NEW: indices of transitions in current round

    # commented out to use standard DQN model to test reward shaping
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
    
    def get_card_probabilities(self, board):
        # Deck composition: eleven 11s, ten 10s, ..., one 1, two jokers
        deck = {i: 12 - i for i in range(1, 12)}  # 11:11, 10:10, ..., 1:1vitality
        deck['joker'] = 2
        total_cards = 68

        # Flatten board and remove zeros (empty slots)
        discarded_cards = [card for card in np.array(board).flatten() if card != 0]
        for card in discarded_cards:
            if card in deck and deck[card] > 0:
                deck[card] -= 1
                total_cards -= 1

        # Calculate probability for each card type
        probabilities = []
        for i in range(1, 12):
            prob = (deck[i] / total_cards) if total_cards > 0 else 0
            probabilities.append(prob)
        # Joker
        probabilities.append(deck['joker'] / total_cards if total_cards > 0 else 0)
        return probabilities

    def get_all_players_card_probabilities(self, hand, board, player_names=None):
        """
        Returns a dictionary mapping each player to their probability array.
        'self' key for agent, others by name (or index if names not provided).
        """
        deck = {i: 12 - i for i in range(1, 12)}
        deck['joker'] = 2
        total_cards = 68

        # Remove discarded cards from deck
        discarded_cards = [card for card in np.array(board).flatten() if card != 0]
        for card in discarded_cards:
            if card in deck and deck[card] > 0:
                deck[card] -= 1
                total_cards -= 1

        # Remove agent's own hand from deck for self-probability
        hand_cards = [card for card in np.array(hand).flatten() if card != 0]
        for card in hand_cards:
            if card in deck and deck[card] > 0:
                deck[card] -= 1
                total_cards -= 1

        # Probabilities for self (agent knows its hand)
        self_probs = [1.0 if hand_cards.count(i) > 0 else 0.0 for i in range(1, 12)]
        self_probs.append(1.0 if hand_cards.count('joker') > 0 else 0.0)

        # Probabilities for other players (based on remaining deck)
        other_probs = []
        for _ in range(3):
            probs = []
            for i in range(1, 12):
                prob = (deck[i] / total_cards) if total_cards > 0 else 0
                probs.append(prob)
            probs.append(deck['joker'] / total_cards if total_cards > 0 else 0)
            other_probs.append(probs)

        # Build dictionary
        prob_dict = {}
        prob_dict['self'] = self_probs
        if player_names is None:
            for idx in range(3):
                prob_dict[f'player_{idx+1}'] = other_probs[idx]
        else:
            for idx, name in enumerate(player_names):
                prob_dict[name] = other_probs[idx]
        return prob_dict

    def get_deck_card_counts(self, hand, board, normalize=True):
        """
        Return a numpy array length 12 with remaining counts for cards 1..11 and joker.
        Index 0 -> card 1, index 10 -> card 11, index 11 -> joker.

        Deck composition (initial): card v has v copies for v in 1..11, and 2 jokers.
        """
        # Initialize deck counts: 1..11 -> counts equal to value, joker -> 2
        deck = {i: i for i in range(1, 12)}
        deck['joker'] = 2

        # Subtract discarded cards from board
        discarded_cards = [c for c in np.array(board).flatten() if c != 0]
        for card in discarded_cards:
            if card in deck and deck[card] > 0:
                deck[card] -= 1

        # Subtract agent's own hand cards
        hand_cards = [c for c in np.array(hand).flatten() if c != 0]
        for card in hand_cards:
            if card in deck and deck[card] > 0:
                deck[card] -= 1

        # Build ordered array: card 1 .. 11, joker
        counts = np.array([float(deck[i]) for i in range(1, 12)] + [float(deck['joker'])], dtype=np.float32)
        if normalize:
            # initial counts: card v has v copies (1..11), joker has 2
            initial = np.array([float(i) for i in range(1, 12)] + [2.0], dtype=np.float32)
            # avoid division by zero (shouldn't happen) but safe-guard
            with np.errstate(divide='ignore', invalid='ignore'):
                counts = np.where(initial > 0, counts / initial, 0.0)
        return counts

    def request_action(self, observations):
        hand = np.array(observations["hand"]).flatten()
        board = np.array(observations["board"]).flatten()
        possible_actions_values = list(observations["possible_actions"])

        # Add card counts for other players
        all_players = sorted(p for p in self.player_card_counts if p != self.name)
        card_counts = [self.player_card_counts.get(p, 0) / 17 for p in all_players]
        while len(card_counts) < 3:
            card_counts.append(0)
        # Build deck counts (12 values) and add to state vector
        deck_counts = self.get_deck_card_counts(hand, board)

        obs = np.concatenate([hand / 13, board / 13, card_counts, deck_counts])
        
        possible_actions_mask = np.zeros(self.action_size, dtype=np.float32)
        valid_action_indices = [
            self.all_actions.index(val) for val in possible_actions_values
        ]
        possible_actions_mask[valid_action_indices] = 1.0

        action_index = self.act(obs, possible_actions_mask, valid_action_indices)
        action_str = self.all_actions[action_index]

        
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
            self.last_custom_reward, attack, defense, vit = self.reward.get_reward(reward_info)
            self.attack.append(attack)
            self.defense.append(defense)    
            self.vitality.append(vit)
            if self.last_custom_reward != -1.0:
                self.match_custom_rewards.append(self.last_custom_reward)
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
            # if hasattr(self, 'last_custom_reward') and self.last_custom_reward == 2 and place == 1:
            #     self.last_custom_reward = 5
            start_time = time.time()
            # self.rewards.append(self.last_custom_reward)  # Store reward for this match
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
            
            self.current_turn_actions = []
        self.current_turn_actions = []
        # self.match_custom_rewards = []  # Removed to preserve reward history
        self.round_transition_indices = []

    def plot_rewards(self, path: str, path_averaged: str, window: int = 10):
        import matplotlib.pyplot as plt
        self.rewards = self.match_custom_rewards  # Use the collected match rewards
        if not hasattr(self, "rewards"):
            print("[plot_positions] Warning: No rewards data to plot.")
            return

        plt.figure()
        x = range(len(self.rewards))
        y = self.rewards
        plt.plot(x, y, label="Attack", linestyle="-", marker="o", alpha=0.8)
        plt.xlabel("Match")
        plt.ylabel("Rewards per match")
        plt.title("Agent Reward per Match")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        import matplotlib.pyplot as plt
        import numpy as np

        if not hasattr(self, "rewards"):
            print("[plot_rewards] Warning: No rewards data to plot.")
            return

        rewards = np.array(self.rewards)
        x = np.arange(len(rewards))

        # Compute rolling average
        if len(rewards) >= window:
            rewards_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            x_avg = np.arange(window - 1, len(rewards))
        else:
            rewards_avg = rewards
            x_avg = x

        plt.cla()
        plt.figure()
        plt.plot(x, rewards, label="Reward (raw)", linestyle="-", marker="o", alpha=0.4)
        plt.plot(
            x_avg, rewards_avg, label=f"Reward (avg {window})", linewidth=2, alpha=0.9
        )

        plt.xlabel("Match")
        plt.ylabel("Rewards per match")
        plt.title("Agent Reward per Match")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_averaged)
        plt.close()

         # Second figure: moving averages of attack, defense, vitality
        metrics = [(getattr(self, 'attack', []), 'Attack', 'red'),
                   (getattr(self, 'defense', []), 'Defense', 'green'),
                   (getattr(self, 'vitality', []), 'Vitality', 'purple')]
        plt.figure(figsize=(10, 5))
        for metric, label, color in metrics:
            arr = np.array(metric)
            if len(arr) >= window:
                moving_avg = np.convolve(arr, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(arr)), moving_avg, color=color, label=f'{label} (avg)')
        plt.xlabel('Round')
        plt.ylabel('Metric (Moving Avg)')
        plt.title('Attack, Defense, Vitality (Moving Avg)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(path.replace('.png', '_metrics.png'))
        plt.close()




