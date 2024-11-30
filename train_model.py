import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque
from minesweeper_game import MinesweeperGame
import time
import pickle

# DQN parameters
board_size = 10
num_mines = 10
num_actions = board_size * board_size  # each cell can be an action
gamma = 0.99  # discount factor
epsilon = 1.0  # exploration rate
epsilon_min = 0.1  # minimum exploration rate
epsilon_decay = 0.995  # decay rate for exploration
learning_rate = 0.001
batch_size = 64
memory_size = 10000
train_episodes = 5000
max_steps_per_episode = 1000

file_run_metadata = f"metadata_{int(time.time())}.csv"

# Replay memory
memory = deque(maxlen=memory_size)

def create_dqn_model(input_shape, num_actions):
    model = tf.keras.Sequential()
    model.add(layers.Reshape((*input_shape, 1), input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation = 'relu', use_bias = True))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation = 'relu', use_bias = True))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation = 'relu', use_bias = True))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation = 'relu', use_bias = True))
    model.add(layers.Conv2D(1, (1, 1), padding='same', activation = 'linear', use_bias = True))
    model.add(layers.Flatten())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    # model.summary()
    # input()
    return model

# Initialize the DQN
input_shape = (board_size, board_size)
models = sorted(os.listdir("models"), key=lambda x: int(x))
start_episode = 0

# Check if there is a saved model
if len(models)>0:
    model_path = models[-1]
    print("Loading saved model...")
    model = tf.keras.models.load_model(f'models/{model_path}/minesweeper_dqn_model.keras')
    
    with open(f"models/{model_path}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            start_episode = metadata["episode"]
            epsilon = metadata["epsilon"]

    # Load the target model
    target_model = tf.keras.models.load_model(f"models/{model_path}/minesweeper_target_model.keras")
    
    # Load replay memory
    replay_memory_path = f"models/{model_path}/replay_memory_episode_{model_path}.pkl"
    if os.path.exists(replay_memory_path):
        with open(replay_memory_path, 'rb') as f:
            memory = pickle.load(f)
    else:
        print("Replay memory not found, starting with an empty memory.")
else:
    model = create_dqn_model(input_shape, num_actions)
    target_model = create_dqn_model(input_shape, num_actions)
    print("No saved model found, creating a new model.")

# Function to get action based on epsilon-greedy policy
def choose_action(state, epsilon, valid_actions):
    if np.random.rand() <= epsilon:
        return random.choice(valid_actions)
    q_values = model.predict(state[np.newaxis], verbose=0)[0]
    masked_q_values = np.full(q_values.shape, -np.inf)
    masked_q_values[valid_actions] = q_values[valid_actions]
    return np.argmax(masked_q_values)


# Function to store experience in memory
def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Function to train the DQN
def train_dqn():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = model.predict(state[np.newaxis], verbose=0)
        if done:
            target[0][action] = reward
        else:
            future_q = np.max(target_model.predict(next_state[np.newaxis], verbose=0))
            target[0][action] = reward + gamma * future_q
        model.fit(state[np.newaxis], target, epochs=1, verbose=0)

# Training loop
for episode in range(start_episode, train_episodes):
    env = MinesweeperGame(board_size, num_mines)
    # Do the first move at the center
    env._uncover_adjacent(board_size//2, board_size//2)
    state = np.array(env.get_observable_state())
    steps = 0
    total_reward = 0
    start_time = time.time()
    while not env.finished and steps < max_steps_per_episode:
        # Get list of valid actions (covered cells)
        valid_actions = [idx for idx, (r, c) in enumerate([(i, j) for i in range(board_size) for j in range(board_size)]) if (r, c) not in env.uncovered]
        action = choose_action(state, epsilon, valid_actions)
        row, col = divmod(action, board_size)

        # Perform action
        env._uncover_adjacent(row, col)
        next_state = np.array(env.get_observable_state())

        # Check for win or loss
        if env.board[row][col] == -1:
            reward = -10
            env.finished = True
        elif len(env.uncovered) == board_size * board_size - num_mines:
            reward = 50
            env.finished = True
        else:
            reward = 1

        store_experience(state, action, reward, next_state, env.finished)
        train_dqn()
        state = next_state
        total_reward += reward
        steps += 1

    # Epsilon decay and logging as before

    # Update target model every few episodes
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())

    # Epsilon decay
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    end_time = time.time()

    print(f"Episode: {episode+1}, Reward: {total_reward}, Steps: {steps}, Epsilon: {epsilon}, Time: {round(end_time - start_time,ndigits=4)}s")
    
    # Save the model and metadata
    if episode % 10 == 0:
        os.mkdir(f"models/{episode+1}")
        model.save(f"models/{episode+1}/minesweeper_dqn_model.keras")
        target_model.save(f"models/{episode+1}/minesweeper_target_model.keras")

        metadata_dict = {
            "episode": episode+1,
            "reward": total_reward,
            "steps": steps,
            "epsilon": epsilon,
            "time": round(end_time - start_time, ndigits=4)
        }
        
        with open(f"models/{episode+1}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata_dict, f)

        # Save replay memory
        with open(f"models/{episode+1}/replay_memory_episode_{episode+1}.pkl", 'wb') as f:
            pickle.dump(memory, f)
    
    with open(file_run_metadata, 'a') as f:
        f.write(f"{episode+1},{total_reward},{steps},{epsilon},{round(end_time - start_time,ndigits=4)}\n")
    

# Save the final model
model.save("minesweeper_dqn_model.keras")
print("Model saved!")
