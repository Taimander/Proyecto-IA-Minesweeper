import pygame
from minesweeper_game import MinesweeperGame

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

    def __init__(self, game):
        self.game = game
        self.tile_size = 50
        self.width = self.game.board_size * self.tile_size
        self.height = self.game.board_size * self.tile_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 36)
    
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
                    x, y = pygame.mouse.get_pos()
                    row = y // self.tile_size
                    col = x // self.tile_size
                    if (row, col) not in self.game.uncovered:
                        self.game._uncover_adjacent(row, col)
                    self.game.get_observable_state()
            self.draw()

if __name__ == '__main__':
    pygame.init()
    game = MinesweeperGame(10, 10,show_board=True)
    renderer = MinesweeperGameRenderer(game)
    renderer.run()
