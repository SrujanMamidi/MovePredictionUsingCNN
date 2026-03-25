import numpy as np
import chess

PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

CASTLING_CHANNELS = {
    'K': 12,
    'Q': 13,
    'k': 14,
    'q': 15,
}

TURN_CHANNEL = 16

def encode_board(board: chess.Board):
    """
    Encodes the board into an 8x8x17 representation.
    Channels:
    0-5: White pieces (P, N, B, R, Q, K)
    6-11: Black pieces (p, n, b, r, q, k)
    12-15: Castling rights (K, Q, k, q)
    16: Side to move (1 if white, 0 if black)
    """
    encoded = np.zeros((8, 8, 17), dtype=np.float32)
    
    # Pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
            encoded[rank, file, channel] = 1
            
    # Castling Rights
    fen_parts = board.fen().split()
    castling_rights = fen_parts[2]
    for char in castling_rights:
        if char in CASTLING_CHANNELS:
            encoded[:, :, CASTLING_CHANNELS[char]] = 1
            
    # Turn
    if board.turn == chess.WHITE:
        encoded[:, :, TURN_CHANNEL] = 1
        
    return encoded

def decode_move(index, board: chess.Board, label_encoder):
    """
    Decodes the predicted index back into a SAN move.
    """
    return label_encoder.inverse_transform([index])[0]

def encode_move(move_san, label_encoder):
    """
    Encodes a SAN move into a numerical index.
    """
    return label_encoder.transform([move_san])[0]
