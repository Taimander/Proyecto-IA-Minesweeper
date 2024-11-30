import cv2
import pyautogui
import numpy as np
import os
import pygame
from minesweeper_game import MinesweeperGame
import tensorflow as tf

BOARD_SIZE = 10

def load_templates():
    templates = []
    for f in os.listdir('imgs'):
        no_path = f.split('/')[-1].split('.')[0]
        templates.append((no_path,cv2.imread(str(f'imgs/{f}'))))
    return templates

def get_screen():
    screen = pyautogui.screenshot(region=(850, 150, 200, 200))
    screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
    return screen

def find_instances(screen, template):
    positions = []
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        positions.append(pt)
    return positions

def recreate_board():
    screen = get_screen()
    templates = load_templates()
    instances = []
    for name, template in templates:
        positions = find_instances(screen, template)
        instances.append((name, positions))
        print(f"{name}: {len(positions)}")
    
    # Put a dot on each instance
    for name, positions in instances:
        for pos in positions:
            cv2.circle(screen, pos, 5, (0, 0, 255), -1)
    # Save result
    for i in range(10):
        for j in range(10):
            cv2.circle(screen, (21+i*16, 17+j*16), 5, (0, 255, 0), -1)
    cv2.imwrite('result.png', screen)

def get_board():
    screen = get_screen()
    templates = load_templates()
    instances = []
    for name, template in templates:
        positions = find_instances(screen, template)
        for pos in positions:
            instances.append((name, pos))
    initial_post = (21,17)
    board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for dx in range(BOARD_SIZE):
        for dy in range(BOARD_SIZE):
            closest = None
            closest_dist = float('inf')
            for name, pos in instances:
                dist = (pos[0] - initial_post[0] - dx*16)**2 + (pos[1] - initial_post[1] - dy*16)**2
                if dist < closest_dist:
                    closest = int(name) if name != 'covered' else -1
                    closest_dist = dist
            board[dy][dx] = closest
    return board
    
###
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

    def __init__(self, model):
        self.tile_size = 50
        self.width = 10 * self.tile_size
        self.height = 10 * self.tile_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 36)
        self.last_circle = None

        self.model = model
    
    def draw(self):
        board = get_board()
        self.screen.fill((255, 255, 255))
        for row in range(10):
            for col in range(10):
                x = col * self.tile_size
                y = row * self.tile_size
                pygame.draw.rect(self.screen, (0, 0, 0), (x, y, self.tile_size, self.tile_size), 1)
                if board[row][col] == -1:
                        # pygame.draw.rect(self.screen, (255, 0, 0), (x, y, self.tile_size, self.tile_size))
                        pass
                else:
                    text = self.font.render(str(board[row][col]), True, self.colors[str(board[row][col])])
                    self.screen.blit(text, (x + self.tile_size // 2 - text.get_width() // 2, y + self.tile_size // 2 - text.get_height() // 2))
        if self.last_circle is not None:
            pygame.draw.circle(self.screen, (0, 0, 255), self.last_circle, 5)
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Get the observable state (as in training)
                    state = np.array(get_board())

                    # Predict Q-values for the current state
                    q_values = self.model.predict(state[np.newaxis], verbose=0)[0]

                    # Mask already uncovered cells
                    for row in range(10):
                        for col in range(10):
                            if state[row][col] != -1:
                                # Set a very low value to prevent the model from choosing uncovered cells
                                q_values[row * 10 + col] = -float('inf')

                    # Choose the action with the highest Q-value (ignoring uncovered cells)
                    action = np.argmax(q_values)
                    row, col = divmod(action, 10)

                    # Print the predicted action
                    print(f"Predicted action: {action}, Row: {row}, Col: {col}")

                    # Draw a circle on the predicted cell
                    x = col * self.tile_size + self.tile_size // 2
                    y = row * self.tile_size + self.tile_size // 2
                    self.last_circle = (x, y)

                    # Only uncover if the cell is not already uncovered
                    # if (row, col) not in self.game.uncovered:
                    #     self.game._uncover_adjacent(row, col)
            self.draw()
            


if __name__ == '__main__':
    pygame.init()
    model = tf.keras.models.load_model('models_tf/1721/minesweeper_dqn_model.keras')
    # game = MinesweeperGame(10, 10, show_board=True)
    renderer = MinesweeperGameRenderer(model)
    renderer.run()