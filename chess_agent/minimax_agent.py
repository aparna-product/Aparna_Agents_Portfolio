import chess
import random
from tqdm import tqdm # Used to show progress for longer calculations

# --- 1. Evaluation Function (How good is a position?) ---
# This function assigns a numerical score to a chess position.
# Positive score means White is better, Negative means Black is better.

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000  # King value is high to avoid getting checked/mated
}

def evaluate_board(board):
    """Simple material evaluation."""
    score = 0
    # Check for game end conditions
    if board.is_checkmate():
        # If it's White's turn and checkmate, Black won.
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_seventyfive_moves():
        return 0

    # Count the value of all pieces
    for piece_type in PIECE_VALUES:
        # Get the number of White's pieces of this type
        white_count = len(board.pieces(piece_type, chess.WHITE))
        # Get the number of Black's pieces of this type
        black_count = len(board.pieces(piece_type, chess.BLACK))
        
        # Add to the total score
        score += PIECE_VALUES[piece_type] * (white_count - black_count)
    
    return score

# --- 2. The Minimax Algorithm (The Brain of the AI) ---
# This is a recursive algorithm that searches for the "best" move by assuming
# the opponent will always play the move that is worst for us (minimize our gain).
# Depth determines how many moves ahead the AI looks.

def minimax(board, depth, is_maximizing_player, alpha, beta):
    """Minimax algorithm with Alpha-Beta Pruning."""
    if depth == 0 or board.is_game_over():
        # Base case: evaluate the board position
        return evaluate_board(board)

    legal_moves = list(board.legal_moves)
    
    if is_maximizing_player: # White's turn (maximize score)
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            # Recursively call minimax for the opponent's turn
            evaluation = minimax(board, depth - 1, False, alpha, beta)
            board.pop() # Undo the move
            
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break # Alpha-Beta Pruning
        return max_eval

    else: # Black's turn (minimize score)
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            # Recursively call minimax for the opponent's turn
            evaluation = minimax(board, depth - 1, True, alpha, beta)
            board.pop() # Undo the move
            
            min_eval = min(min_eval, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break # Alpha-Beta Pruning
        return min_eval

# --- 3. The Move Selector (Chooses the Best Move) ---

def find_best_move(board, depth, is_white_agent):
    """Finds the best move by iterating through all legal moves and using minimax."""
    best_move = None
    
    # White wants to maximize the score, Black wants to minimize it.
    best_eval = -float('inf') if is_white_agent else float('inf')
    
    # Use tqdm to show a progress bar in the terminal
    print(f"\nAgent calculating best move at depth {depth}...")
    for move in tqdm(list(board.legal_moves)):
        board.push(move)
        # Check the resulting board's value (start with the opponent's turn)
        evaluation = minimax(board, depth - 1, not is_white_agent, -float('inf'), float('inf'))
        board.pop() # Undo the move
        
        if is_white_agent:
            if evaluation > best_eval:
                best_eval = evaluation
                best_move = move
        else: # Black Agent
            if evaluation < best_eval:
                best_eval = evaluation
                best_move = move
                
    # If no move was found (e.g., in a checkmate position), pick a random legal move
    if best_move is None and list(board.legal_moves):
        best_move = random.choice(list(board.legal_moves))
        
    return best_move

# --- 4. Game Setup and Loop ---

def run_ai_match(white_depth, black_depth):
    """Sets up and runs the chess match between the two AI agents."""
    board = chess.Board()
    print("--- 🤖 AI Chess Match Started! 🤖 ---")
    
    move_count = 0
    
    # The main game loop
    while not board.is_game_over():
        move_count += 1
        
        # Determine whose turn it is
        is_white_turn = board.turn == chess.WHITE
        current_color = "White" if is_white_turn else "Black"
        current_depth = white_depth if is_white_turn else black_depth
        
        print(f"\n--- Move {move_count}: {current_color}'s Turn (Depth: {current_depth}) ---")
        
        # Get the move from the respective AI agent
        best_move = find_best_move(board, current_depth, is_white_turn)
        
        if best_move:
            # Execute the move on the board
            board.push(best_move)
            print(f"{current_color} plays: {best_move.uci()}")
            # Print the board in text format (ASCII)
            print(board)
            # You can also visualize the board (if running in a Jupyter/IPython environment)
            # print(board._repr_svg_()) # If you want to see a nice SVG image
        else:
            # This should only happen if a game over condition was detected earlier
            break
            
    # Game Over
    print("\n--- Game Over! ---")
    print(f"Result: {board.result()}")
    print(f"Total Moves: {move_count}")

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # You can change these values to make one agent 'smarter' than the other!
    # A depth of 3 is a good starting point for simple agents.
    WHITE_AGENT_DEPTH = 3 
    BLACK_AGENT_DEPTH = 3 
    
    run_ai_match(WHITE_AGENT_DEPTH, BLACK_AGENT_DEPTH)
