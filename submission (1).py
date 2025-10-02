
import numpy as np

def is_valid_location(board, col, configuration):
    """Checks if a column is a valid location to drop a piece."""
    # A column is valid if the top cell (index 0 in that column) is empty (0)
    return board[col] == 0

def drop_piece(board, col, piece, configuration):
    """
    Drops a piece into the specified column on a copy of the board.
    Returns the new board state.
    """
    new_board = board.copy()
    # Find the lowest empty row in the selected column
    for row in range(configuration['rows'] - 1, -1, -1):
        if new_board[row * configuration['columns'] + col] == 0:
            new_board[row * configuration['columns'] + col] = piece
            return new_board
    # Should not reach here if is_valid_location is checked before calling
    return board

def check_win(board, piece, configuration):
    """Checks if the given piece has won the game."""
    rows = configuration['rows']
    columns = configuration['columns']
    inarow = configuration['inarow']

    # Convert the 1D board list to a 2D numpy array for easier indexing
    board_array = np.array(board).reshape(rows, columns)

    # Check horizontal win
    for r in range(rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r, c + i] == piece for i in range(inarow)):
                return True

    # Check vertical win
    for c in range(columns):
        for r in range(rows - inarow + 1):
            if all(board_array[r + i, c] == piece for i in range(inarow)):
                return True

    # Check positively sloped diagonals
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            if all(board_array[r + i, c + i] == piece for i in range(inarow)):
                return True

    # Check negatively sloped diagonals
    for r in range(inarow - 1, rows):
        for c in range(columns - inarow + 1):
            if all(board_array[r - i, c + i] == piece for i in range(inarow)):
                return True

    return False

def evaluate_window(window, piece, configuration):
    """
    Evaluates the score of a window (list) of cells for a given piece.
    Assigns scores based on potential winning lines.
    """
    score = 0
    opponent_piece = 1 if piece == 2 else 2
    inarow = configuration['inarow']

    # High score for a winning window
    if window.count(piece) == inarow:
        score += 100
    # Good score for a potential winning line with one empty spot
    elif window.count(piece) == inarow - 1 and window.count(0) == 1:
        score += 5
    # Decent score for a potential winning line with two empty spots
    elif window.count(piece) == inarow - 2 and window.count(0) == 2:
        score += 2

    # Penalize the opponent having a potential winning line with one empty spot
    if window.count(opponent_piece) == inarow - 1 and window.count(0) == 1:
        score -= 4

    return score

def score_position(board, piece, configuration):
    """
    Evaluates the score of the entire board for a given piece.
    Considers horizontal, vertical, diagonal wins and center control.
    """
    score = 0
    rows = configuration['rows']
    columns = configuration['columns']
    inarow = configuration['inarow']
    board_array = np.array(board).reshape(rows, columns)

    # Score center column (strategic advantage)
    center_array = [int(i) for i in list(board_array[:, columns // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score Horizontal windows
    for r in range(rows):
        row_array = [int(i) for i in list(board_array[r, :])]
        for c in range(columns - inarow + 1):
            window = row_array[c:c + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score Vertical windows
    for c in range(columns):
        col_array = [int(i) for i in list(board_array[:, c])]
        for r in range(rows - inarow + 1):
            window = col_array[r:r + inarow]
            score += evaluate_window(window, piece, configuration)

    # Score positive sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    # Score negative sloped diagonal windows
    for r in range(rows - inarow + 1):
        for c in range(columns - inarow + 1):
            window = [board_array[r + inarow - 1 - i, c + i] for i in range(inarow)]
            score += evaluate_window(window, piece, configuration)

    return score

def is_terminal_node(board, configuration):
    """Checks if the current board state is terminal (win, loss, or draw)."""
    # Check if either player has won or if the board is full
    return check_win(board, 1, configuration) or check_win(board, 2, configuration) or all(cell != 0 for cell in board)

def minimax(board, depth, alpha, beta, maximizingPlayer, configuration):
    """
    Minimax algorithm with Alpha-Beta Pruning to find the optimal move.

    Args:
        board (list): The current game board (1D list).
        depth (int): The current depth of the search tree.
        alpha (float): The best value found so far for the maximizing player.
        beta (float): The best value found so far for the minimizing player.
        maximizingPlayer (bool): True if it's the maximizing player's turn, False otherwise.
        configuration (dict): Game configuration.

    Returns:
        tuple: A tuple containing the best column (int) and the corresponding score (float).
               Returns (None, score) for terminal or depth-limited nodes.
    """
    # Get a list of columns where a piece can be dropped
    valid_locations = [col for col in range(configuration['columns']) if is_valid_location(board, col, configuration)]
    # Check if the current node is a game-ending state
    is_terminal = is_terminal_node(board, configuration)

    # If the game is over or the search depth is reached
    if is_terminal or depth == 0:
        if is_terminal:
            if check_win(board, 1, configuration): # Player 1 wins (maximizing player)
                return (None, 100000000000000) # Return a very high score
            elif check_win(board, 2, configuration): # Player 2 wins (minimizing player)
                return (None, -10000000000000) # Return a very low score
            else: # Game is a draw
                return (None, 0) # Return a neutral score
        else: # Depth is 0 (reached the search limit)
            # Evaluate the position heuristically from the perspective of the maximizing player
            # Note: The score_position function evaluates based on player 1's pieces.
            # We adjust the score interpretation based on whose turn it is in minimax.
            return (None, score_position(board, 1, configuration) if maximizingPlayer else -score_position(board, 2, configuration))


    # If it's the maximizing player's turn
    if maximizingPlayer:
        value = -np.inf # Initialize the best value for the maximizing player
        column = np.random.choice(valid_locations) # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Note: The drop_piece function is not strictly needed here if we find the row correctly
            # We just need the index to update the board copy.
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration['rows'] - 1, -1, -1) if b_copy[r * configuration['columns'] + col] == 0)
            b_copy[row * configuration['columns'] + col] = 1 # Drop player 1's piece (maximizing)

            # Recursively call minimax for the minimizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, False, configuration)[1]

            # Update the best value and corresponding column for the maximizing player
            if new_score > value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the maximizing player
            # is greater than or equal to the best value found so far for the minimizing player (beta),
            # the minimizing player will avoid this branch, so we can prune it.
            alpha = max(alpha, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

    # If it's the minimizing player's turn
    else: # Minimizing player
        value = np.inf # Initialize the best value for the minimizing player
        column = np.random.choice(valid_locations) # Initialize with a random valid move (fallback)

        # Iterate through all valid moves
        for col in valid_locations:
            # Create a copy of the board and drop the piece
            b_copy = board.copy()
            # Find the correct row to drop the piece
            row = next(r for r in range(configuration['rows'] - 1, -1, -1) if b_copy[r * configuration['columns'] + col] == 0)
            b_copy[row * configuration['columns'] + col] = 2 # Drop player 2's piece (minimizing)

            # Recursively call minimax for the maximizing player's turn
            new_score = minimax(b_copy, depth - 1, alpha, beta, True, configuration)[1]

            # Update the best value and corresponding column for the minimizing player
            # The minimizing player wants to minimize the score from the maximizing player's perspective
            if new_score < value:
                value = new_score
                column = col

            # Alpha-Beta Pruning: If the current best value for the minimizing player
            # is less than or equal to the best value found so far for the maximizing player (alpha),
            # the maximizing player will avoid this branch, so we can prune it.
            beta = min(beta, value)
            if alpha >= beta:
                break # Prune the remaining branches

        return column, value

def minimax_agent(observation, configuration, search_depth=3):
    """
    ConnectX agent that uses the Minimax algorithm with Alpha-Beta Pruning to choose a move.

    Args:
        observation (dict): A dictionary containing the game observation, including:
            - 'board' (list): A 1D list representing the game board (0: empty, 1: Player 1, 2: Player 2).
            - 'mark' (int): The current player's mark (1 or 2).
        configuration (dict): A dictionary containing the game configuration, including:
            - 'columns' (int): The number of columns on the board.
            - 'rows' (int): The number of rows on the board.
            - 'inarow' (int): The number of checkers in a row required to win.
        search_depth (int): The maximum depth the minimax algorithm will search.

    Returns:
        int: The chosen column to drop a checker.
    """
    board = observation.board
    player = observation.mark

    # Call the minimax function to find the best move using the specified search depth.
    col, minimax_score = minimax(board, search_depth, -np.inf, np.inf, player == 1, configuration)

    # Ensure the chosen column is valid before returning.
    valid_locations = [col for col in range(configuration['columns']) if is_valid_location(board, col, configuration)]
    if col is None or col not in valid_locations:
        # If minimax doesn't return a valid move (shouldn't happen with sufficient depth and correct implementation)
        # or the returned column is somehow invalid, choose the first valid column as a fallback.
        return valid_locations[0]

    return int(col)
