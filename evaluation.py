import numpy as np

import game_functions as gf


def evaluate_state(board: np.ndarray) -> float:
    """
    Returns the score of the given board state.
    :param board: The board state for which the score is to be calculated.
    :return: The score of the given board state.
    """
    # TODO: Complete evaluate_state function to return a score for the current state of the board
    # Hint: You may need to use the np.nonzero function to find the indices of non-zero elements.
    # Hint: You may need to use the gf.within_bounds function to check if a position is within the bounds of the board.

    positions = np.transpose(np.nonzero(board))
    weight1 = 10
    weight2 = 500

    score = weight1*monotony(board, positions) + weight2*smoothness(board)
    return score

    # raise NotImplementedError("Evaluation function not implemented yet.")


def monotony(board, positions):

    score = 0
    rows, cols = 4, 4
    weight_arr = [[0 for i in range(cols)] for j in range(rows)]

    for i in range(rows):
        if [i, 0] in positions:
            weight_arr[i][0] = (5-i)*200
    for i in range(rows):
        for j in range(cols):
            score += weight_arr[i][j]*board[i, j]

    return score


def smoothness(board):

    score = 0

    for i in range(4):
        for j in range(4):
            if (i+1 < 4):
                score += -1*abs(board[i, j]-board[i+1, j])
            if (j+1 < 4):
                score += -1*abs(board[i, j]-board[i, j+1])

    return score
