import pandas as pd
import numpy as np
import chess
import chess.svg
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import gradio as gr

# ---------------- Load Model ----------------
model = load_model('az_chess_model.h5')
label_encoder = joblib.load('label_encoder.joblib')

# ---------------- Board Encoding ----------------
PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}

CASTLING_TO_CHANNEL = {'K':12,'Q':13,'k':14,'q':15}
TURN_CHANNEL = 16


def encode_board_cnn(board):
    encoded_board = np.zeros((8,8,17), dtype=np.float32)

    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            rank, file = i//8, i%8
            channel = PIECE_TO_CHANNEL[(piece.piece_type,piece.color)]
            encoded_board[rank,file,channel] = 1

    castling = board.fen().split(' ')[2]
    for c in castling:
        if c in CASTLING_TO_CHANNEL:
            encoded_board[:,:,CASTLING_TO_CHANNEL[c]] = 1

    if board.turn == chess.WHITE:
        encoded_board[:,:,TURN_CHANNEL] = 1

    return encoded_board


def decode_best_legal_move_from_probs(board, probs):

    encoder_count = len(label_encoder.classes_)
    model_count = int(probs.shape[0])
    usable = min(encoder_count, model_count)

    ranked = np.argsort(probs[:usable])[::-1]

    for idx in ranked:
        move = label_encoder.inverse_transform([int(idx)])[0]

        test_board = board.copy()
        try:
            test_board.push_san(move)
            return move, encoder_count, model_count
        except:
            continue

    return None, encoder_count, model_count


# ---------------- Prediction Function ----------------
def predict_and_display(fen_string):

    try:
        board = chess.Board(fen_string)
    except:
        return "Invalid FEN", "", ""

    encoded = np.expand_dims(encode_board_cnn(board), axis=0)

    policy_probs, value_score = model.predict(encoded, verbose=0)

    move, encoder_count, model_count = decode_best_legal_move_from_probs(board, policy_probs[0])

    value = value_score[0][0]

    try:
        board.push_san(move)
    except:
        pass

    svg = chess.svg.board(board=board, size=450)

    value_text = f"{value:.4f}  (White > 0 | Black < 0)"

    return move, value_text, svg


# ---------------- UI Styling ----------------

css = """
.gradio-container {
    background:
        radial-gradient(circle at 10% 20%, rgba(255, 215, 120, 0.25), transparent 30%),
        radial-gradient(circle at 90% 10%, rgba(120, 180, 255, 0.2), transparent 32%),
        linear-gradient(145deg, #f6efe6 0%, #e5edf7 55%, #f9f5ee 100%);
    min-height: 100vh;
}

#app-title {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    letter-spacing: 0.4px;
    color: #1c2432;
    margin: 6px 0 4px 0;
}

#app-subtitle {
    text-align: center;
    color: #3f4a5e;
    margin-bottom: 18px;
}

#left-panel,
#right-panel {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid rgba(40, 50, 70, 0.12);
    border-radius: 18px;
    box-shadow: 0 16px 34px rgba(25, 32, 46, 0.14);
    padding: 18px;
}

#fen-box textarea {
    background: #0f1724 !important;
    color: #f3f6ff !important;
    border: 1px solid #2d4268 !important;
    border-radius: 10px !important;
    font-family: "Courier New", monospace !important;
    font-size: 14px !important;
    line-height: 1.45 !important;
}

#fen-box label {
    font-weight: 700 !important;
    color: #223047 !important;
}

#predict-btn {
    border-radius: 10px !important;
    border: 0 !important;
    background: linear-gradient(135deg, #2d5ed7, #4f8df3) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    min-height: 42px;
}

#predict-btn:hover {
    filter: brightness(1.05);
}

#move-box textarea,
#value-box textarea {
    background: #f7f9fe !important;
    border: 1px solid #d3deef !important;
    border-radius: 10px !important;
    color: #1f2a3d !important;
    font-weight: 600 !important;
}

#board-view {
    border: 1px solid #d4deed;
    border-radius: 12px;
    background: #ffffff;
    padding: 8px;
}
"""

start_fen = chess.STARTING_FEN


# ---------------- Gradio UI ----------------

with gr.Blocks(
        title="AI Chess Predictor",
        theme=gr.themes.Soft(),
        css=css
) as demo:

    gr.Markdown("<div id='app-title'>Chess Move Predictor</div>")
    gr.Markdown("<div id='app-subtitle'>Enter a FEN position to predict the next move and evaluate the board.</div>")

    with gr.Row():

        with gr.Column(scale=1, elem_id="left-panel"):
            fen_input = gr.Textbox(
                label="Board Position (FEN)",
                value=start_fen,
                lines=2,
                elem_id="fen-box"
            )

            predict_btn = gr.Button("Predict Move", variant="primary", elem_id="predict-btn")

            move_output = gr.Textbox(label="Predicted Move", elem_id="move-box")

            value_output = gr.Textbox(label="Position Evaluation", elem_id="value-box")

        with gr.Column(scale=1, elem_id="right-panel"):
            board_output = gr.HTML(label="Chess Board", elem_id="board-view")

    predict_btn.click(
        predict_and_display,
        inputs=fen_input,
        outputs=[move_output, value_output, board_output]
    )


demo.launch(share=True)

