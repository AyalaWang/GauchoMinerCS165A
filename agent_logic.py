import numpy as np
import os

from constants import *

class Agent:
    def __init__(self):
        """Initialize the agent with weights and step counter from checkpoint or defaults."""
        self.weights, self.step_counter = self.load_checkpoint()
        self.prev_features = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_game_over = None
        self.current_features = None
        self.current_action = None
        self.max_q_value = None

    def initialize_weights(self):
        """Random weight initialization."""
        return np.random.randn(FEATURE_DIM, NUM_ACTIONS) * 0.01

    def load_checkpoint(self):
        """Load checkpoint or initialize weights."""
        if os.path.exists(CHECKPOINT_PATH):
            try:
                checkpoint = np.load(CHECKPOINT_PATH, allow_pickle=False)
                weights = checkpoint.get('weights')
                if weights is None:
                    weights = self.initialize_weights()
                return weights, int(checkpoint['step_counter'])
            except Exception:
                pass
        return self.initialize_weights(), 0

    def save_checkpoint(self, force=False):
        """Save weights and step count."""
        if force or self.step_counter % SAVE_INTERVAL == 0:
            try:
                np.savez(CHECKPOINT_PATH, weights=self.weights, step_counter=self.step_counter)
            except Exception:
                pass

    def encode_binary(self, value, num_bits):
        """Efficient binary encoding using numpy."""
        value = max(0, int(min(value, 2**num_bits - 1)))  # Clamp value to valid range
        bits = np.unpackbits(np.array([value], dtype=f'>u{num_bits // 8}').view(np.uint8))
        return bits[-num_bits:]

    def extract_features(self, local_map, position, energy, score, gold_count):
        """Efficient feature vector extraction."""
        features = np.zeros(FEATURE_DIM, dtype=np.uint8)
        features[-1] = 1  # bias

        # Map features (one-hot encode 9x9xNUM_BLOCK_TYPES)
        flat_map = local_map.flatten()
        indices = NUM_BLOCK_TYPES * np.arange(flat_map.size) + flat_map
        features[indices] = 1

        offset = BASE_FEATURE_DIM

        # Energy (16-bit)
        energy_bits = self.encode_binary(energy, 16)
        features[offset:offset + 16] = energy_bits
        offset += 16

        # Score (16-bit)
        score_bits = self.encode_binary(score, 16)
        features[offset:offset + 16] = score_bits
        offset += 16

        # Gold count (8 bits per direction × 4)
        for direction in ['W', 'A', 'S', 'D']:
            gold_bits = self.encode_binary(gold_count.get(direction, 0), 8)
            features[offset:offset + 8] = gold_bits
            offset += 8

        return features

    def calculate_q_value(self, features, action):
        return np.dot(self.weights[:, action], features)

    def get_epsilon(self):
        """Linearly decaying epsilon."""
        decay_rate = (EPSILON_START - EPSILON_END) / DECAY_STEPS
        return max(EPSILON_END, EPSILON_START - decay_rate * self.step_counter)

    def agent_logic(self, local_map, position, energy, score, gold_count, training):
        self.current_features = self.extract_features(local_map, position, energy, score, gold_count)
        q_values = np.dot(self.weights.T, self.current_features)
        epsilon = self.get_epsilon()

        if training and np.random.rand() < epsilon:
            self.current_action = np.random.choice(NUM_ACTIONS)
        else:
            self.current_action = int(np.argmax(q_values))

        return ACTION_TO_CHAR[self.current_action]

    def update_q_learning(self, delta_energy, delta_score, game_over):
        """Q-learning weight update."""
        if self.prev_features is not None:
            assert None not in [self.prev_action, self.prev_reward, self.prev_game_over]

            prev_q = np.dot(self.weights[:, self.prev_action], self.prev_features)
            next_q_values = np.dot(self.weights.T, self.current_features)
            max_next_q = 0 if self.prev_game_over else np.max(next_q_values)

            target = self.prev_reward + DISCOUNT_FACTOR * max_next_q
            td_error = target - prev_q

            delta_w = LEARNING_RATE * td_error * self.prev_features
            delta_w = np.clip(delta_w, -0.01, 0.01)
            self.weights[:, self.prev_action] += delta_w

        reward = 1 * delta_score + 0.15 * delta_energy

        self.prev_features = self.current_features
        self.prev_action = self.current_action
        self.prev_reward = reward
        self.prev_game_over = game_over

        self.step_counter += 1
        self.save_checkpoint()