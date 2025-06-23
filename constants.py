
# Blocks
VOID = 0
EMPTY = 1
DIRT = 2
STONE = 3
DEEPSLATE = 4
STONE_GOLD = 5
DEEPSLATE_GOLD = 6
CHEST = 7
BARREL = 8
CREEPER = 9
ZOMBIE = 10
SKELETON = 11

BLOCK_TYPES = [
    VOID,
    EMPTY,
    DIRT,
    STONE,
    DEEPSLATE,
    STONE_GOLD,
    DEEPSLATE_GOLD,
    CHEST,
    BARREL,
    CREEPER,
    ZOMBIE,
    SKELETON,
]
NUM_BLOCK_TYPES = len(BLOCK_TYPES)

# Mapping from block ID to name (used in reward/energy lookups)
BLOCK_ID_TO_NAME = {
    VOID: 'VOID',
    EMPTY: 'EMPTY',
    DIRT: 'DIRT',
    STONE: 'STONE',
    DEEPSLATE: 'DEEPSLATE',
    STONE_GOLD: 'GOLD_ORE',
    DEEPSLATE_GOLD: 'DEEPSLATE_GOLD_ORE',
    CHEST: 'CHEST',
    BARREL: 'BARREL',
    CREEPER: 'CREEPER',
    ZOMBIE: 'ZOMBIE',
    SKELETON: 'SKELETON',
}

# Energy cost/gain for each block type
BLOCK_ENERGY = {
    'VOID': -1,
    'EMPTY': -1,
    'DIRT': -2,
    'STONE': -4,
    'DEEPSLATE': -10,
    'STONE_GOLD': -4,
    'DEEPSLATE_GOLD': -10,
    'ZOMBIE': -50,
    'SKELETON': -1,
    'CREEPER': -400,
    'CHEST': -1,
    'BARREL': 80,  # Barrel gives energy instead of taking it
}

# Reward values for each block type
BLOCK_REWARD = {
    'VOID': 0,
    'EMPTY': 0,
    'DIRT': 0,
    'STONE': 0,
    'DEEPSLATE': 0,
    'STONE_GOLD': 5,
    'DEEPSLATE_GOLD': 5,
    'ZOMBIE': 10,
    'SKELETON': 10,
    'CREEPER': 20,
    'CHEST': 10,
    'BARREL': 0,
}

# Actions
ACTIONS = [
    (0, -1),  # W = 0 (up)
    (-1, 0),  # A = 1 (left)
    (0, 1),   # S = 2 (down)
    (1, 0),   # D = 3 (right)
    (0, 0)    # I = 4 (idle)
]
NUM_ACTIONS = len(ACTIONS)
ACTION_TO_CHAR = ['W', 'A', 'S', 'D', 'I']

# Features
VIEW_WIDTH = 9
VIEW_HEIGHT = 9
ENERGY_BITS = 16
SCORE_BITS = 16
GOLD_COUNT_BITS = 8 * 4
BASE_FEATURE_DIM = VIEW_WIDTH * VIEW_HEIGHT * NUM_BLOCK_TYPES
FEATURE_DIM = BASE_FEATURE_DIM + ENERGY_BITS + SCORE_BITS + GOLD_COUNT_BITS + 1

# Misc
CHECKPOINT_PATH = 'ckpt.npz'

# Training arguments
SAVE_INTERVAL = 100000

# Q-Learning constants
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.05
DECAY_STEPS = 1000000
