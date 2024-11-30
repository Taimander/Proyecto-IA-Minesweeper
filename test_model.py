from minesweeper_game import MinesweeperGame
import numpy as np
import tensorflow as tf
import multiprocessing

GAMES_TO_PLAY = 10000

def simulateGame(model):
    game = MinesweeperGame(10, 10, show_board=False)
    while not game.finished:
        state = np.array(game.get_observable_state())
        q_values = model.predict(state[np.newaxis], verbose=0)[0]
        # Mask already uncovered cells
        for row in range(game.board_size):
            for col in range(game.board_size):
                if (row, col) in game.uncovered:
                    # Set a very low value to prevent the model from choosing uncovered cells
                    q_values[row * game.board_size + col] = -float('inf')
        action = np.argmax(q_values)
        row, col = divmod(action, game.board_size)
        # Only uncover if the cell is not already uncovered
        if (row, col) not in game.uncovered:
            game._uncover_adjacent(row, col)
    return game.result

def process(model, amt):
    wins = 0
    for _ in range(amt):
        wins += simulateGame(model)
    return wins

if __name__=='__main__':
    model = tf.keras.models.load_model('models/1681/minesweeper_dqn_model.keras')
    wins = 0
    with multiprocessing.Pool() as pool:
        wins = sum(pool.starmap(process, [(model, GAMES_TO_PLAY//10) for _ in range(10)]))
    print(f"Wins: {wins}/{GAMES_TO_PLAY}")
    print(f"Win rate: {wins*100/GAMES_TO_PLAY:.2f}%")
