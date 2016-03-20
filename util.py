import math
import numpy as np


TILES = {(245, 149, 99): 16, (237, 194, 46): 2048, 
(242, 177, 121): 8, (237, 204, 97): 256, 
(246, 124, 95): 32, (237, 197, 63): 1024, 
(237, 200, 80): 512, (237, 207, 114): 128, 
(246, 94, 59): 64, (238, 228, 218): 2, 
(237, 224, 200): 4, (205, 193, 180): 0}

def get_reward(score_change):
    if score_change == 0:
        return 0
    else: 
        return score_change / 16

def normalize_num(x):
    """
    We assume the max value that can appear in the grid is 2^16 = 65536. 
    It is possible that 2^17 occurs but even with optimal strategy, it is very unlikely. 
    
    Some example numbers are mapped as follows:

    0     -> 0
    2     -> log(2)/16 = 1/16
    4     -> log(4)/16 = 1/8
    2048  -> log(2048)/16 = 11/16
    65536 -> log(65536)/16 = 1
    """
    if x == 0:
        return 1/18.0

    return math.log(x, 2)/18.0 + 1/18.0

def normalize(board):
    """Takes in the board and normalizes the values to serve as input to the Q-Network.

    Args:
        board: 4x4 vector representing the board
    Returns:
        normalized: 4x4 array representing the normalized board    
    """

    normalized = []
    for i in range (4):
        normalized.append(map(normalize_num, board[i]))

    return normalized

def flatten(board):
    """Takes 4x4 list and flattens it a (16) numpy vector"""
    vec = np.array(board, dtype="float")
    return np.reshape(vec, (16))

