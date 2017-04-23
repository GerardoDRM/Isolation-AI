
# Build a Game-playing Agent

![Example game of isolation](viz.gif)

## Synopsis
This project implements basics AI techniques on adversarial games; in this case
I develop the minimax algorithm, minimax with alpha-beta pruning and 3 heuristics or
evaluation function to improve results from given example by Udacity.

The 3 heuristics are described on heuristics_analysis document.

## Quickstart Guide

The following example creates a game and illustrates the basic API. You can run this example with `python sample_players.py`

    from isolation import Board

    # create an isolation board (by default 7x7)
    player1 = RandomPlayer()
    player2 = GreedyPlayer()
    game = Board(player1, player2)

    # place player 1 on the board at row 2, column 3, then place player 2 on
    # the board at row 0, column 5; display the resulting board state.  Note
    # that .apply_move() changes the calling object
    game.apply_move((2, 3))
    game.apply_move((0, 5))
    print(game.to_string())

    # players take turns moving on the board, so player1 should be next to move
    assert(player1 == game.active_player)

    # get a list of the legal moves available to the active player
    print(game.get_legal_moves())

    # get a successor of the current state by making a copy of the board and
    # applying a move. Notice that this does NOT change the calling object
    # (unlike .apply_move()).
    new_game = game.forecast_move((1, 1))
    assert(new_game.to_string() != game.to_string())
    print("\nOld state:\n{}".format(game.to_string()))
    print("\nNew state:\n{}".format(new_game.to_string()))

    # play the remainder of the game automatically -- outcome can be "illegal
    # move" or "timeout"; it should _always_ be "illegal move" in this example
    winner, history, outcome = game.play()
    print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
    print(game.to_string())
    print("Move history:\n{!s}".format(history))


### Tournament

The `tournament.py` script is used to evaluate the effectiveness of your custom_score heuristic.  The script measures relative performance of your agent (called "Student") in a round-robin tournament against several other pre-defined agents.  The Student agent uses time-limited Iterative Deepening and the custom_score heuristic you wrote.

The performance of time-limited iterative deepening search is hardware dependent (faster hardware is expected to search deeper than slower hardware in the same amount of time).  The script controls for these effects by also measuring the baseline performance of an agent called "ID_Improved" that uess Iterative Deepening and the improved_score heuristic from `sample_players.py`.  Your goal is to develop a heuristic such that Student outperforms ID_Improved.

The tournament opponents are listed below. (See also: sample heuristics and players defined in sample_players.py)

- Random: An agent that randomly chooses a move each turn.
- MM_Null: CustomPlayer agent using fixed-depth minimax search and the null_score heuristic
- MM_Open: CustomPlayer agent using fixed-depth minimax search and the open_move_score heuristic
- MM_Improved: CustomPlayer agent using fixed-depth minimax search and the improved_score heuristic
- AB_Null: CustomPlayer agent using fixed-depth alpha-beta search and the null_score heuristic
- AB_Open: CustomPlayer agent using fixed-depth alpha-beta search and the open_move_score heuristic
- AB_Improved: CustomPlayer agent using fixed-depth alpha-beta search and the improved_score heuristic


## Using the Board Visualization

The `isoviz` folder contains a modified version of chessboard.js that can animate games played on a 7x7 board.  In order to use the board, you must run a local webserver by running `python -m SimpleHTTPServer 8000` from your project directory (you can replace 8000 with another port number if that one is unavailable), then open your browser to `http://localhost:8000` and navigate to the `/isoviz/display.html` page.  Enter the move history of an isolation match (i.e., the array returned by the Board.play() method) into the text area and run the match.  Refresh the page to run a different game.
