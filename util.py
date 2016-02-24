import numpy as np
import math

TILES = {(245, 149, 99): 16, (237, 194, 46): 2048, 
(242, 177, 121): 8, (237, 204, 97): 256, 
(246, 124, 95): 32, (237, 197, 63): 1024, 
(237, 200, 80): 512, (237, 207, 114): 128, 
(246, 94, 59): 64, (238, 228, 218): 2, 
(237, 224, 200): 4, (205, 193, 180): 0}


def normalize(board):
    """Takes in the board and normalizes the values to serve as input to the Q-Network.

    We assume the max value that can appear in the grid is 2^16 = 65536. 
    It is possible that 2^17 occurs but even with optimal strategy, it is very unlikely. 
    
    Some example array values are mapped as follows:

    0     -> 0
    2     -> log(2)/16 = 1/16
    4     -> log(4)/16 = 1/8
    2048  -> log(2048)/16 = 11/16
    65536 -> log(65536)/16 = 1

    Args:
        board: 4x4 vector representing the board
    Returns:
        normalized: 4x4 array representing the normalized board    
    """

    normalized = []
    for i in range (4):
        normalized.append(map(lambda x: 0 if x == 0 else math.log(x, 2)/16, board[i]))

    return normalized

def flatten(board):
    vec = np.array(board, dtype="float")
    return np.reshape(vec, (1, 16))

