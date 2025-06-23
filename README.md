# GauchoMinerCS165A
Implemented a Q-learning agent for a grid-based mining game. The agent learns to mine valuable blocks efficiently while managing energy and avoiding hazards, using feature-based states and temporal-difference updates. 

1. Main logic is in agent_logic.py
2. In order to run the game make sure you have Python installed and run the game using python3 new_game.py or python new_game.py (depending on which python version you use)

This command starts the game with default settings. You can customize the game environment
using command-line arguments to adjust map generation, game mechanics, and display settings. Key
arguments include:
• --seed: Sets the random seed for map generation (default: 42). Use the same seed to generate
identical maps.
• --width, --height: Define the grid dimensions (default: 50x30).
• --p_gold, --p_dgold: Set the probability of gold in stone (default: 0.2) and deepslate (default:
0.4).
• --zombies, --creepers, --skeletons, --chests, --barrels: Specify the number of each entity
(default: 20 zombies, 10 creepers, 10 skeletons, 15 chests, 15 barrels).
• --energy: Sets the miner’s initial energy (default: 1000).
• --training: Enables (1) or disables (0) training mode (default: 1). We will only call the
update_q_learning function in training mode. If you want to evaluate your implementation,
please set it to 0.
• --fps: Controls game speed in frames per second (default: 5). Lower values (e.g., 1) slow the
visualization for easier debugging.
• --fog: Enables (1) or disables (0) fog of war, limiting visibility to a 9x9 area (default: 1). Note:
Disabling fog is for debugging only; the agent still cannot see the entire map.
• --grid_size: Sets the size of each grid cell in pixels (default: 24).
• --render: Enables (1) or disables (0) graphical rendering (default: 1). Disable rendering for faster
training or testing

3. You can train the model using the command python3 training.py --episodes 10000 --fps 10000 --save_interval 100
To train your Q-learning agent, run the command python training.py from the terminal. You can
customize the training process by modifying training.py to suit your specific needs. The script accepts
the following command-line arguments:
• --episodes: Sets the number of training episodes (default: 10000). Higher values enable the agent
to learn from more diverse experiences.
• --fps: Controls the game speed in frames per second during training (default: 1000).
• --save_interval: Specifies the frequency, in episodes, for saving model checkpoints and printing
training statistics (default: 100).
