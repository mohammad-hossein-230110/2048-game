import numpy as np

import evaluation
import game_functions as gf


class MCTS:
    def __init__(self, board, mode='ucb'):
        # You may change this parameter to scale the exploration term in the UCB formula.
        self.C_CONSTANT = 2
        # You may change this parameter to scale the depth to which the agent searches.
        self.SD_SCALE_PARAM = 8
        # You may change this parameter to scale the depth to which the agent searches.
        self.TM_SCALE_PARAM = 8
        # You may change this parameter to scale the depth to which the agent searches.
        self.SCALER_PARAM = 200
        # You may change this parameter to scale the depth to which the agent searches.
        self.UCB_SD_SCALE_PARAM = 10
        # You may change this parameter to scale the depth to which the agent searches.
        self.UCB_TM_SCALE_PARAM = 20
        # You may change this parameter to scale the depth to which the agent searches.
        self.UCB_SCALER_PARAM = 300
        self.board = board
        self.mode = mode

    def get_search_params(self, move_number: int) -> (int, int):
        """
        Returns the depth to which the agent should search for the given move number.
        ...
        :type move_number: int
        :param move_number: The current move number.
        :return: The depth to which the agent should search for the given move number.
        """
        # TODO: Complete get_search_params function to return the depth to which the agent should search for the given move number.
        # Hint: You may want to use the self.SD_SCALE_PARAM, self.SL_SCALE_PARAM, and self.SCALER_PARAM parameters.
        # Hint: You may want to use the self.UCB_SPM_SCALE_PARAM, self.UCB_SL_SCALE_PARAM, and self.UCB_SCALER_PARAM parameters.
        # Hint: You may want to use the self.mode parameter to check which mode the agent is on.

        add = move_number//self.SCALER_PARAM

        if self.mode == 'mcts':
            return self.SD_SCALE_PARAM + add, self.TM_SCALE_PARAM + add
        else:
            return self.UCB_SD_SCALE_PARAM + add, self.UCB_TM_SCALE_PARAM + add

    def ai_move(self, board, move_number):
        search_depth, total_moves = self.get_search_params(move_number)
        if self.mode == 'ucb':
            best_move = self.mcts_v2(board, total_moves * 4, search_depth)
        else:
            best_move = self.mcts_v0(board, total_moves, search_depth)
        return best_move

    @staticmethod
    def simulate_move(board: np.ndarray, search_depth: int) -> float:
        """
        Returns the score of the given board state.
        :param board: The board state for which the score is to be calculated.
        :param search_depth: The depth to which the agent should search for the given board state.
        :return: The score of the given board state.
        """
        # TODO: Complete simulate_move function to simulate a move and return the score of the given board state.
        # Hint: You may want to use the gf.random_move function to simulate a random move.
        # Hint: You may want to use the evaluation.evaluate_state function to score a board.
        # Hint: You may want to use the move_made returned from the gf.random_move function to check if a move was made.
        # Hint: You may want to use the gf.add_new_tile function to add a new tile to the board.

        sum = 0

        for i in range(search_depth):

            board_copy = np.copy(board)

            board, move_made, score = gf.random_move(board_copy)

            if move_made:

                board_copy = np.copy(board)

                board = gf.add_new_tile(board_copy)

                sum += score

        return sum

    def mcts_v0(self, board: np.ndarray, total_moves: int, search_depth: int):
        """
        Returns the best move for the given board state.
        ...
        :type search_depth: int
        :type total_moves: int
        :type board: np.ndarray
        :param board: The board state for which the best move is to be found.
        :param total_moves: The total number of moves to be simulated.
        :param search_depth: The depth to which the agent should search for the given board state.
        :return: Returns the best move for the given board state.
        """
        # TODO: Complete mcts_v0 function to return the best move for the given board state.
        # Hint: You may want to use the gf.get_moves function to get all possible moves.
        # Hint: You may want to use the gf.add_new_tile function to add a new tile to the board.
        # Hint: You may want to use the self.simulate_move function to simulate a move.
        # Hint: You may want to use the np.argmax function to get the index of the maximum value in an array.
        # Hint: You may want to use the np.zeros function to create an array of zeros.
        # Hint: You may want to use the np.copy function to create a copy of a numpy array.

        actions = gf.get_moves()
        best_action = None
        max = 0
        for action in actions:
            new_board = np.copy(board)
            new_board, move_made, score = action(new_board)
            if move_made:
                addition_score = 0
                for i in range(total_moves):
                    addition_score = addition_score*i
                    addition_score += self.simulate_move(
                        np.copy(new_board), search_depth)
                    addition_score = addition_score/(i+1)
                    score += addition_score
                    if score > max:
                        max = score
                        best_action = action
        return best_action

        # raise NotImplementedError("MCTS v0 not implemented yet.")
    def ucb(self, moves: list, total_visits: int) -> np.ndarray:
        """
        Returns the UCB scores for the given moves.
        :param moves: The moves for which the UCB scores are to be calculated.
        :param total_visits: The total number of visits for all moves.
        :return: The UCB scores for the given moves.
        """
        # TODO: Complete ucb function to return the UCB scores for the given moves.
        # Hint: You may want to use the self.C_CONSTANT parameter to scale the exploration term in the UCB formula.
        # Hint: You may want to use np.inf to represent infinity.
        # Hint: You may want to use np.sqrt to calculate the square root of a number.
        # Hint: You may want to use np.log to calculate the natural logarithm of a number.

        infinity = np.inf

        scores = []

        for i in range(len(moves)):
            action = moves[i][0]
            score = moves[i][1]
            repeat_num = moves[i][2]
            if repeat_num == 0:
                scores.append((action, infinity))
                continue
            calculate = (score/repeat_num) + self.C_CONSTANT * \
                (np.sqrt(np.log(total_visits)/repeat_num))
            scores.append((action, calculate))
        return scores

    def mcts_v2(self, board, total_moves, search_depth):
        """
        Returns the best move for the given board state.
        ...
        :type search_depth: int
        :type total_moves: int
        :type board: np.ndarray
        :param board: The board state for which the best move is to be found.
        :param total_moves: The total number of moves to be simulated.
        :param search_depth: The depth to which the agent should search for the given board state.
        :return: Returns the best move for the given board state.
        """
        # TODO: Complete mcts_v2 function to return the best move for the given board state.
        # Hint: You may want to use the gf.get_moves function to get all possible moves.
        # Hint: You may want to use the gf.add_new_tile function to add a new tile to the board.
        # Hint: You may want to use the self.simulate_move function to simulate a move.
        # Hint: You may want to use the np.argmax function to get the index of the maximum value in an array.
        # Hint: You may want to use the np.copy function to create a copy of a numpy array.
        # Hint: You may want to use the self.ucb function to get the UCB scores for the given moves.

        visit = 0
        actions = gf.get_moves()
        pos = [x[1] for x in gf.get_all_possible_moves(board)]

        moves = [(actions[pos[i]], 0, 0) for i in range(len(pos))]
        for i in range(total_moves):

            ucb_results = self.ucb(moves, visit)

            ucb_results = np.array(ucb_results)
            moves = np.array(moves)
            index = np.argmax(ucb_results[:, 1])

            new_action = ucb_results[index, 0]

            new_board, move_made, score = new_action(np.copy(board))

            visit = visit+1

            moves[index, 1] = moves[index, 1] * moves[index, 2]

            moves[index, 1] = moves[index, 1] + score + \
                self.simulate_move(new_board, search_depth)

            moves[index, 2] = moves[index, 2]+1

            moves[index, 1] = moves[index, 1] / moves[index, 2]

        index = np.argmax(moves[:, 1])
        final_action = ucb_results[index, 0]
        return final_action
