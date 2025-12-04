import chess
import pandas as pd
import numpy as np

# Funcion que me devuelve el estado del tablero a partir de los movimientos
def board_after_n_moves(moves_str, n_moves=None):
    board = chess.Board()
    moves = moves_str.split()
    
    if n_moves is not None:
        moves = moves[:n_moves]

    for mv in moves:
        try:
            board.push_san(mv)  # SAN = notación estándar que tú tienes
        except:
            break  # por si algún movimiento está mal anotado

    return board


piece_values = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0
}

# Funcion que me devuelve las nuevas variables creadas. (nº de piezas, valor de las piezas)
def material_features(board):
    white_total = 0
    black_total = 0
    white_count = 0
    black_count = 0
    
    for piece in board.piece_map().values():
        value = piece_values[piece.piece_type]
        
        if piece.color == chess.WHITE:
            white_total += value
            white_count += 1
        else:
            black_total += value
            black_count += 1
    
    return {
        "white_piece_count": white_count,
        "black_piece_count": black_count,
        "white_material_value": white_total,
        "black_material_value": black_total,
        "material_balance": white_total - black_total
    }

CENTER_SQUARES = [
    chess.D4, chess.E4, chess.D5, chess.E5
]

# Funcion que devuelve los valores para evaluar el control del centro del tablero
def center_control(board):    
    white = 0
    black = 0
    
    for sq in CENTER_SQUARES: 
        attackers_white = board.attackers(chess.WHITE, sq) # Devuelve los atacantes de una casilla específica por color
        attackers_black = board.attackers(chess.BLACK, sq)
        
        if attackers_white:
            white += 1
        if attackers_black:
            black += 1
            
    return {
        "white_center_control": white,
        "black_center_control": black
    }

PST = {
    chess.PAWN: np.array([
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10,-20,-20, 10, 10, 5,
        5, -5,-10, 0, 0,-10, -5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5,10,25,25,10,5,5,
        10,10,20,30,30,20,10,10,
        50,50,50,50,50,50,50,50,
        0,0,0,0,0,0,0,0
    ]),
    chess.KNIGHT: np.array([
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20, 0, 0, 0, 0,-20,-40,
        -30, 0,10,15,15,10,0,-30,
        -30, 5,15,20,20,15,5,-30,
        -30, 0,15,20,20,15,0,-30,
        -30, 5,10,15,15,10,5,-30,
        -40,-20, 0,5,5,0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]),
    chess.BISHOP: np.zeros(64),
    chess.ROOK: np.zeros(64),
    chess.QUEEN: np.zeros(64),
    chess.KING: np.zeros(64),
}


# Más robusto: generar tablero con turn=color y contar legal_moves
def mobility_for_color(board, color):
    b2 = board.copy()
    b2.turn = color
    return b2.legal_moves.count()


# PST score (suma simples)
def pst_score(board, color):
    total = 0
    for sq, p in board.piece_map().items():
        val = PST.get(p.piece_type)
        if val is None:
            continue
        index = sq
        if p.color == chess.BLACK:
            # invertir para perspectiva black
            index = chess.square_mirror(index)
        total += val[index] * (1 if p.color==color else -1)
    return total if color==chess.WHITE else -total


# función agregadora: nuevas features posicionales
def positional_features(board):
    feats = {}
    feats.update({
        "white_mobility": mobility_for_color(board, chess.WHITE),
        "black_mobility": mobility_for_color(board, chess.BLACK),
        "white_PST": pst_score(board, chess.WHITE),
        "black_PST": pst_score(board, chess.BLACK),
    })
    # espacio relativo: casillas controladas en la mitad rival
    white_half = [sq for sq in chess.SQUARES if chess.square_rank(sq) >= 4]  # ranks 5-8 => 4..7
    black_half = [sq for sq in chess.SQUARES if chess.square_rank(sq) <= 3]  # ranks 1-4 => 0..3
    feats["white_space"] = sum(1 for sq in white_half if board.attackers(chess.WHITE, sq))
    feats["black_space"] = sum(1 for sq in black_half if board.attackers(chess.BLACK, sq))
    return feats

def extract_features_from_moves(moves_str, n_moves=None):
    board = board_after_n_moves(moves_str, n_moves)
    
    feats = {}
    feats.update(material_features(board))
    feats.update(center_control(board))
    feats.update(positional_features(board))
    
    return feats

def add_move_features(df,n_moves=None):
    feature_rows = []

    if not hasattr(n_moves, "__iter__"):
        n_moves = pd.Series([n_moves] * len(df), index=df.index)

    for moves, n in zip(df["moves"], n_moves):
        if pd.isna(n):
            feature_rows.append(extract_features_from_moves(moves, None))
        else:
            feature_rows.append(extract_features_from_moves(moves, int(n)))

    feature_df = pd.DataFrame(feature_rows)
    return pd.concat([df, feature_df], axis=1)

def count_plies(moves_str):
    if pd.isna(moves_str):
        return 0
    return len(moves_str.split())