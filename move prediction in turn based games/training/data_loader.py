import pandas as pd
import numpy as np
import chess
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from utils.encoding import encode_board

def load_and_preprocess_data(csv_path, sample_size=5000):
    df = pd.read_csv(csv_path)
    if 'moves' not in df.columns:
        raise ValueError("Dataset must contain a 'moves' column")
        
    moves_data = df['moves'].head(sample_size)
    positions = []
    labels = []
    
    # We also need values for the value head. 
    # Since the lichess dataset doesn't have Stockfish evals, 
    # we can use the game winner as a proxy (1 for win, 0 for draw, -1 for loss).
    winners = df['winner'].head(sample_size)
    values = []

    for idx, (moves_sequence, winner) in enumerate(zip(moves_data, winners)):
        if not isinstance(moves_sequence, str):
            continue
            
        board = chess.Board()
        move_list = moves_sequence.split()
        
        # Result for value head
        if winner == 'white':
            result = 1.0
        elif winner == 'black':
            result = -1.0
        else:
            result = 0.0

        for move_san in move_list:
            # Store current board and the move played
            positions.append(encode_board(board))
            labels.append(move_san)
            values.append(result)
            
            try:
                board.push_san(move_san)
            except:
                # If illegal move in sequence, stop processing this game
                positions.pop()
                labels.pop()
                values.pop()
                break
                
    return np.array(positions), np.array(labels), np.array(values)

def get_label_encoder(labels, save_path='label_encoder.joblib'):
    encoder = LabelEncoder()
    encoder.fit(labels)
    joblib.dump(encoder, save_path)
    return encoder
