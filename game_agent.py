"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


"""
     myMoves and opMoves are respectively the amount of possible valid movements
     from the position of the computer and opponent at any one state.
     filledSpaces is the number of spaces on the board that are filled.
     The constant 3 was determined through testing to be the optimal combination in early games.
"""


def heuristic1(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    filled = (game.width * game.height) - len(game.get_blank_spaces())

    return float((my_moves - 3 * opponent_moves) * filled)

'''
    Check the distance to the center to get an advantage on position
'''


def heuristic2(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    walls = [
        [(0, i) for i in range(game.width)],
        [(i, 0) for i in range(game.height)],
        [(game.width - 1, i) for i in range(game.width)],
        [(i, game.height - 1) for i in range(game.height)]]

    # Remove moves that are touching the walls
    my_moves = [m for m in game.get_legal_moves(player) if m not in walls]
    opponent_moves = [m for m in game.get_legal_moves(game.get_opponent(player)) if m not in walls]

    my_moves_score = 0
    opponent_moves_score = 0
    # middle
    cx, cy = int(game.width / 2), int(game.height / 2)
    # Get Manhatan distance on each point that is not touching the walls to the center point
    for m in my_moves:
        p_x, p_y = m
        my_moves_score += abs(p_x - cx) + abs(p_y - cy)

    for m in opponent_moves:
        opp_x, opp_y = m
        opponent_moves_score += abs(opp_x - cx) + abs(opp_y - cy)

    return float(my_moves_score - 2 * opponent_moves_score)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    walls = [
        [(0, i) for i in range(game.width)],
        [(i, 0) for i in range(game.height)],
        [(game.width - 1, i) for i in range(game.width)],
        [(i, game.height - 1) for i in range(game.height)]]

    # Get percentage of ocuppied board
    board = int(len(game.get_blank_spaces()) / (game.width * game.height) * 100)

    if board <= 30:
        my_moves = len(game.get_legal_moves(player))
        opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

        return float((my_moves - 2 * opponent_moves))
    elif board > 30  and board <= 50:
        # Avoid walls
        # Remove moves that are on the walls
        my_moves = [m for m in game.get_legal_moves(player) if m not in walls]
        opponent_moves = [m for m in game.get_legal_moves(game.get_opponent(player)) if m not in walls]

        return float(len(my_moves) - len(opponent_moves))
    else:
        # Getting players position to create a partition
        p_x, p_y = game.get_player_location(player)
        opp_x, opp_y = game.get_player_location(game.get_opponent(player))

        blank_spaces = game.get_blank_spaces()

        # Check partition area
        # Get the midpoint to determinate the best area to generate the partition
        midpoint_x, midpoint_y = int((p_x + opp_x) / 2), int((p_y + opp_y) / 2)
        # Get players moves
        my_moves = game.get_legal_moves(player)
        opponent_moves = game.get_legal_moves(game.get_opponent(player))

        # Check if horizontal or vertical partition can be applied
        if p_x != opp_x:  # vertical
            # Remove moves that are out of the partition
            if midpoint_x < p_x:  # Rigth
                my_moves = [m for m in my_moves if m[0] > midpoint_x]
                opponent_moves = [m for m in opponent_moves if m[0] < midpoint_x]
            else:  # left
                my_moves = [m for m in my_moves if m[0] < midpoint_x]
                opponent_moves = [m for m in opponent_moves if m[0] > midpoint_x]

        else:  # horizontal
            # Remove moves that are out of the partition
            if midpoint_y < p_y:  # Up
                my_moves = [m for m in my_moves if m[1] > midpoint_y]
                opponent_moves = [m for m in opponent_moves if m[1] < midpoint_y]
            else:  # Down
                my_moves = [m for m in my_moves if m[1] < midpoint_y]
                opponent_moves = [m for m in opponent_moves if m[1] > midpoint_y]

        return float(len(my_moves) + len(opponent_moves))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if not legal_moves:
            return (-1, -1)

        best_move = None
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                depth = 1
                # Try until timeout and get best move searching on a deepest
                # level
                while True:
                    if self.method == "minimax":
                        _, mv = self.minimax(game, depth)
                        best_move = mv
                    elif self.method == "alphabeta":
                        _, mv = self.alphabeta(game, depth)
                        best_move = mv
                    depth += 1
            else:
                if self.method == "minimax":
                    _, best_move = self.minimax(game, self.search_depth)

                elif self.method == "alphabeta":
                    _, best_move = self.alphabeta(game, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Get legal moves for active player
        l_moves = game.get_legal_moves()

        # Check if we iterate to last level or if we found final state
        if depth == 0 or game.utility == float("inf") or game.utility == float("-inf"):
            # Heuristic on node
            return self.score(game, self), (-1, -1)

        l_moves = game.get_legal_moves()
        # maximizing player
        if maximizing_player:
            best_value, best_move = float("-inf"), (-1, -1)
            for m in l_moves:
                possible_state = game.forecast_move(m)
                v, _ = self.minimax(possible_state, depth - 1, False)
                if v > best_value:
                    best_value, best_move = v, m

            return best_value, best_move
        # minimizing player
        else:
            best_value, best_move = float("inf"), (-1, -1)
            for m in l_moves:
                possible_state = game.forecast_move(m)
                v, _ = self.minimax(possible_state, depth - 1, True)
                if v < best_value:
                    best_value, best_move = v, m
            return best_value, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Get legal moves for active player
        l_moves = game.get_legal_moves()

        # Check if we iterate to last level or if we found final state
        if depth == 0 or game.utility == float("inf") or game.utility == float("-inf"):
            # Heuristic on node
            return self.score(game, self), (-1, -1)

        l_moves = game.get_legal_moves()
        # maximizing player
        if maximizing_player:
            best_value, best_move = float("-inf"), (-1, -1)
            for m in l_moves:
                possible_state = game.forecast_move(m)
                v, _ = self.alphabeta(
                    possible_state, depth - 1, alpha, beta, False)
                if v > best_value:
                    best_value, best_move = v, m
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    best_value, best_move = alpha, m
                    break

            return best_value, best_move
        # minimizing player
        else:
            best_value, best_move = float("inf"), (-1, -1)
            for m in l_moves:
                possible_state = game.forecast_move(m)
                v, _ = self.alphabeta(
                    possible_state, depth - 1, alpha, beta, True)
                if v < best_value:
                    best_value, best_move = v, m
                beta = min(best_value, beta)
                if beta <= alpha:
                    best_value, best_move = beta, m
                    break

            return best_value, best_move
