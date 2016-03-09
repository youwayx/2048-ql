"""
Implementation of 2048 based off of https://github.com/gabrielecirulli/2048

"""
import random

class Game:

    def __init__(self):
        self.grid = [[0,0,0,0] for i in range (4)]
        self.score = 0
        self.vector_dict = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def available_cells(self):
        """Returns the coordinates of all the empty cells."""
        a = []
        for i in range (4):
            for j in range (4):
                if self.grid[i][j] == 0:
                    a.append((i,j))

        return a

    def add_random_tile(self):
        """
        Adds a random tile based on the available tiles. The random tile
        has a value of 2 with 0.9 probability and a value of 4 with 0.1 
        probability.
        """
        avail_cells = self.available_cells()
        if len(avail_cells) == 0:
            return False

        index = random.randint(0, len(avail_cells) - 1)
        row, col = avail_cells[index]
        epsilon = random.random()

        if epsilon < 0.9:
            self.grid[row][col] = 2
        else:
            self.grid[row][col] = 4
        return True

    def move(self, direction):
        """
        Move tiles in direction, merging and updating score if necessary.
        0: up
        1: right
        2: down
        3: left
        """

        vector = self.vector_dict[direction]
        traversals = self.get_traversals(vector)

        moved = False  # boolean to track whether any tile has moved
        for i in range (4):
            merged = False  # only one merge per row/col
            for j in range (4):
                row, col = traversals[i*4 + j]
                tile_val = self.grid[row][col]
                if tile_val == 0:
                    continue

                new_row, new_col = self.new_tile_position(row, col, vector)
                if new_row != row or new_col != col:
                    moved = True

                self.grid[row][col] = 0
                if self.grid[new_row][new_col] == tile_val and not merged:
                    self.grid[new_row][new_col] = tile_val * 2
                    self.score += tile_val*2
                else:
                    self.grid[new_row][new_col] = tile_val

        return moved

    def in_bounds(self, x, y):
        """Checks if the x, y are within grid boundaries."""
        if x < 0 or x > 3 or y < 0 or y > 3:
            return False
        return True

    def new_tile_position(self, r, c, vector):
        """
        Given the row, column and movement vector of a tile, return the  
        new row and new column that the tile should be moved to.
        Returns the original row and column if the tile cannot be moved.
        """
        val = self.grid[r][c]
        row = r
        col = c
        new_row = r
        new_col = c
        while True:
            new_row += vector[0]
            new_col += vector[1]
            if not self.in_bounds(new_row, new_col):
                break
            if self.grid[new_row][new_col] == val:
                return [new_row, new_col]
            if self.grid[new_row][new_col] > 0:
                break

            row = new_row
            col = new_col

        return [row, col]

    def get_traversals(self, vector):
        """
        Gets a list of 16 coordinates corresponding to positions on the
        board which determine the order in which movement should be processed.
        """
        xs = [i for i in range (4)]
        ys = [i for i in range (4)]

        tuples = []
        
        # up and down
        if vector[0] != 0:
            if vector[0] == 1:
                ys = ys[::-1]
            for i in range (4):
                for j in range (4):
                    tuples.append((ys[j], xs[i]))
        
        # left and right
        if vector[1] != 0:
            if vector[1] == 1:
                xs = xs[::-1]
            for i in range (4):
                for j in range (4):
                    tuples.append((ys[i], xs[j]))

        return tuples

    def display(self):
        """Prints the board."""
        for b in self.grid:
            print b



## Uncomment below to play the game in the terminal.

# game = Game()
# while (True):
#     if not game.add_random_tile():
#         print "YOU LOSE"
#         break
#     game.display()
#     while True:
#         move = input()
#         if game.move(move):
#             break
#         else:
#             print "invalid move"
#             game.display()
