"""
Implementation of 2048 based off of https://github.com/gabrielecirulli/2048

"""
import random

class Game:

    def __init__(self):
        self.grid = [[0,0,0,0] for i in range (4)]
        self.score = 0
        self.vector_dict = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

    def available_cells(self):
        a = []
        for i in range (4):
            for j in range (4):
                if self.grid[i][j] == 0:
                    a.append((i,j))

        return a

    def add_random_tile(self):
        avail_cells = self.available_cells()

        index = random.randint(0, len(avail_cells))
        row, col = avail_cells[index]
        epsilon = random.random()

        if epsilon < 0.9:
            self.grid[row][col] = 2
        else:
            self.grid[row][col] = 4

    def move(self, direction):
        """
        move tiles in direction, merging if necessary
        0: up
        1: right
        2: down
        3: left
        """

        vector = self.vector_dict[direction]
        traversals = self.get_traversals(vector)

        for row in traversals[0]:
            merged = False  # only one merge per row
            for col in traversals[1]:
                tile_val = self.grid[row][col]
                if tile_val == 0:
                    continue

                new_r, new_c = self.move_tile(row, col, vector)

                if self.in_bounds(new_r, new_c) and self.grid[new_r][new_c] == next_coord \
                    and not merged:
                    self.grid[new_r][new_c] = tile_val * 2
                    self.grid[row][col] = 0
                    self.score += tile_val*2

    def in_bounds(self, x, y):
        if x < 0 or x > 4 or y < 0 or y > 4:
            return False
        return True

    def move_tile(self, r, c, vector):
        row = r
        col = c
        new_row = r
        new_col = c
        while True:
            new_row += vector[0]
            new_col += vector[1]
            if not self.in_bounds(new_row, new_col):
                break
            if self.grid[new_row][new_col] > 0:
                break

            self.grid[new_row][new_col] = self.grid[row][col]
            self.grid[row][col] = 0
            row = new_row
            col = new_col

        next_coord = [new_row, new_col]
        return next_coord

    def get_traversals(self, vector):
        xs = [i for i in range (4)]
        ys = [i for i in range (4)]

        if vector[0] == 1:
            xs = xs[::-1]
        if vector[1] == 1:
            ys = ys[::-1]

        return [xs, ys]

    def display(self):
        for b in self.grid:
            print b

game = Game()
while (True):
    game.add_random_tile()
    game.display()
    move = input()
    game.move(move)


