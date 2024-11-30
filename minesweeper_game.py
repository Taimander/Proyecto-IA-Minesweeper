import random

class MinesweeperGame:
    def __init__(self, board_size, mines, show_board=False):
        self.board_size = board_size
        self.mines = mines
        self.board = [[0 for _ in range(board_size)] for _ in range(board_size)]
        self.uncovered = set()
        self.finished = False
        self._place_mines()
        self._calculate_numbers()
        self.show_board = show_board
        self.result = None
    
    def _place_mines(self):
        mines = 0
        while mines < self.mines:
            row = random.randint(0, self.board_size - 1)
            col = random.randint(0, self.board_size - 1)
            if self.board[row][col] != -1:
                self.board[row][col] = -1
                mines += 1
    
    def _calculate_numbers(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] != -1:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if 0 <= row + dr < self.board_size and 0 <= col + dc < self.board_size:
                                if self.board[row + dr][col + dc] == -1:
                                    self.board[row][col] += 1
    
    def _uncover_adjacent(self, row, col):
        if self.finished:
            return
        if (row, col) in self.uncovered:
            return
        if len(self.uncovered)==0 and self.board[row][col]==-1:
            self.board[row][col] = 0
            new_row = random.randint(0, self.board_size - 1)
            new_col = random.randint(0, self.board_size - 1)
            while self.board[new_row][new_col] == -1 or (new_row, new_col) == (row, col):
                new_row = random.randint(0, self.board_size - 1)
                new_col = random.randint(0, self.board_size - 1)
            self.board[new_row][new_col] = -1
            self._calculate_numbers()
        self.uncovered.add((row, col))
        if self.board[row][col] == 0:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if 0 <= row + dr < self.board_size and 0 <= col + dc < self.board_size:
                        self._uncover_adjacent(row + dr, col + dc)
        if self.board[row][col] == -1:
            self.finished = True
            print("LOSE!")
            self.result = 0
        else:
            self.check_if_win()

    def check_if_win(self):
        if len(self.uncovered) == self.board_size ** 2 - self.mines:
            self.finished = True
            print("WIN!")
            self.result = 1

    def get_observable_state(self):
        observable_board = [[-1 for _ in range(self.board_size)] for _ in range(self.board_size)]
        for row, col in self.uncovered:
            observable_board[row][col] = self.board[row][col]
        if self.show_board:
            for row in range(self.board_size):
                for col in range(self.board_size):
                    print('-' if observable_board[row][col]==-1 else observable_board[row][col], end=' ')
                print()
            print()
        return observable_board
