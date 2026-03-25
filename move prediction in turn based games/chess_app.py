import streamlit as st
import chess
import chess.svg
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# --- Use Streamlit's cache to load the model only once ---
@st.cache_resource
def load_my_model():
    """Loads the ML model and encoder, caching them for performance."""
    try:
        model = load_model('az_chess_model.h5')
        encoder = joblib.load('label_encoder.joblib')
        return model, encoder
    except Exception as e:
        # Display an error in the app if files are missing
        st.error(f"Error loading model or encoder: {e}")
        st.error("Please ensure 'az_chess_model.h5' and 'label_encoder.joblib' are in the same directory.")
        return None, None

# --- MISSING PIECE 1: ADD THE ENCODING DICTIONARIES ---
PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}
CASTLING_TO_CHANNEL = {'K': 12, 'Q': 13, 'k': 14, 'q': 15}
TURN_CHANNEL = 16

# --- MISSING PIECE 2: ADD THE ENCODING FUNCTION ---
def encode_board_cnn(board):
    """Encodes a board object into the 8x8x17 format for the CNN."""
    encoded_board = np.zeros((8, 8, 17), dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            rank, file = i // 8, i % 8
            channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
            encoded_board[rank, file, channel] = 1
    castling_fen = board.fen().split(' ')[2]
    for char in castling_fen:
        if char in CASTLING_TO_CHANNEL:
            channel = CASTLING_TO_CHANNEL[char]
            encoded_board[:, :, channel] = 1
    if board.turn == chess.WHITE:
        encoded_board[:, :, TURN_CHANNEL] = 1
    return encoded_board


# --- Main App Logic ---
st.set_page_config(page_title="AI Chess Predictor", layout="wide")
st.title("♟️ AI Chess Move Predictor with Streamlit")

model, label_encoder = load_my_model()

if model and label_encoder:
    # --- App Layout ---
    col1, col2 = st.columns(2)

    with col1:
        st.header("Board Position")
        # Use session state to remember the board across reruns
        if 'fen' not in st.session_state:
            st.session_state.fen = chess.STARTING_FEN

        fen_input = st.text_input("Enter FEN String:", value=st.session_state.fen)
        
        try:
            board = chess.Board(fen_input)
            st.session_state.fen = board.fen()
            board_svg = chess.svg.board(board=board, size=400)
            # Use st.html to render the SVG
            st.html(board_svg)
        except (ValueError, IndexError):
            st.error("Invalid FEN string.")

    with col2:
        st.header("AI Prediction")
        if st.button("Predict Next Move"):
            if 'fen' in st.session_state:
                # Create board object from the session state FEN
                current_board = chess.Board(st.session_state.fen)
                
                # --- THIS LINE NOW WORKS ---
                encoded_board = np.expand_dims(encode_board_cnn(current_board), axis=0)
                
                policy_probs, value_score = model.predict(encoded_board, verbose=0)
                best_move_index = np.argmax(policy_probs[0])
                predicted_move = label_encoder.inverse_transform([best_move_index])[0]
                position_value = value_score[0][0]

                st.metric(label="Predicted Best Move", value=predicted_move)
                st.metric(label="Position Evaluation", value=f"{position_value:.4f}")
                st.info("Positive values favor White, negative values favor Black.")