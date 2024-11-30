import pygame
from minesweeper_game import MinesweeperGame
import numpy as np
import tensorflow as tf


class MinesweeperGameRenderer:
    colors = {
        '0': (50, 50, 50),
        '1': (0, 0, 255),
        '2': (0, 128, 0),
        '3': (255, 0, 0),
        '4': (0, 0, 128),
        '5': (128, 0, 0),
        '6': (0, 128, 128),
        '7': (0, 0, 0),
        '8': (128, 128, 128),
    }

    def __init__(self, game, model):
        self.game = game
        self.tile_size = 50
        self.width = self.game.board_size * self.tile_size
        self.height = self.game.board_size * self.tile_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 36)

        self.model = model
    
    def draw(self):
        self.screen.fill((255, 255, 255))
        for row in range(self.game.board_size):
            for col in range(self.game.board_size):
                x = col * self.tile_size
                y = row * self.tile_size
                pygame.draw.rect(self.screen, (0, 0, 0), (x, y, self.tile_size, self.tile_size), 1)
                if (row, col) in self.game.uncovered:
                    if self.game.board[row][col] == -1:
                        pygame.draw.rect(self.screen, (255, 0, 0), (x, y, self.tile_size, self.tile_size))
                    else:
                        text = self.font.render(str(self.game.board[row][col]), True, self.colors[str(self.game.board[row][col])])
                        self.screen.blit(text, (x + self.tile_size // 2 - text.get_width() // 2, y + self.tile_size // 2 - text.get_height() // 2))
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Get the observable state (as in training)
                    state = np.array(self.game.get_observable_state())

                    # Predict Q-values for the current state
                    q_values = self.model.predict(state[np.newaxis], verbose=0)[0]

                    # Mask already uncovered cells
                    for row in range(self.game.board_size):
                        for col in range(self.game.board_size):
                            if (row, col) in self.game.uncovered:
                                # Set a very low value to prevent the model from choosing uncovered cells
                                q_values[row * self.game.board_size + col] = -float('inf')

                    # Choose the action with the highest Q-value (ignoring uncovered cells)
                    action = np.argmax(q_values)
                    row, col = divmod(action, self.game.board_size)

                    # Print the predicted action
                    print(f"Predicted action: {action}, Row: {row}, Col: {col}")

                    # Only uncover if the cell is not already uncovered
                    if (row, col) not in self.game.uncovered:
                        self.game._uncover_adjacent(row, col)
            self.draw()


if __name__ == '__main__':
    pygame.init()
    model = tf.keras.models.load_model('models_tf/1721/minesweeper_dqn_model.keras')
    game = MinesweeperGame(10, 10, show_board=True)
    renderer = MinesweeperGameRenderer(game, model)
    renderer.run()
