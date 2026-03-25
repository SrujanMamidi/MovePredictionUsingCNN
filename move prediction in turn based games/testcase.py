import numpy as np
import pandas as pd
import chess
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib

# ----------------------------------------------------
# 1. Load model & encoder
# ----------------------------------------------------
model = load_model("az_chess_model.h5")
label_encoder = joblib.load("label_encoder.joblib")

# ----------------------------------------------------
# 2. Load test dataset
# CSV columns required: fen, best_move, value
# ----------------------------------------------------
test_df = pd.read_csv(r"C:\Users\divya\chess\assets\games.csv")

# ----------------------------------------------------
# 3. Board Encoding (reuse your existing function)
# ----------------------------------------------------
PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}
CASTLING_TO_CHANNEL = {'K': 12, 'Q': 13, 'k': 14, 'q': 15}
TURN_CHANNEL = 16

def encode_board_cnn(board):
    encoded = np.zeros((8, 8, 17), dtype=np.float32)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            r, f = sq // 8, sq % 8
            encoded[r, f, PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]] = 1
    castling = board.fen().split(" ")[2]
    for c in castling:
        if c in CASTLING_TO_CHANNEL:
            encoded[:, :, CASTLING_TO_CHANNEL[c]] = 1
    if board.turn == chess.WHITE:
        encoded[:, :, TURN_CHANNEL] = 1
    return encoded

# ----------------------------------------------------
# 4. Prepare test data
# ----------------------------------------------------
X, y_policy_true, y_value_true = [], [], []

for _, row in test_df.iterrows():
    board = chess.Board(row["fen"])
    X.append(encode_board_cnn(board))
    y_policy_true.append(label_encoder.transform([row["best_move"]])[0])
    y_value_true.append(row["value"])

X = np.array(X, dtype=np.float32)
y_policy_true = np.array(y_policy_true)
y_value_true = np.array(y_value_true)

# ----------------------------------------------------
# 5. Model predictions
# ----------------------------------------------------
policy_probs, value_preds = model.predict(X, batch_size=64, verbose=1)
value_preds = value_preds.squeeze()
policy_top1 = np.argmax(policy_probs, axis=1)

# ----------------------------------------------------
# 6. Policy Analysis (Top-K + Illegal Moves)
# ----------------------------------------------------
def in_top_k(prob, true_idx, k):
    return true_idx in np.argsort(prob)[-k:]

top1_hits, top3_hits, top5_hits = [], [], []
illegal_moves = 0

for i, row in test_df.iterrows():
    top1_hits.append(policy_top1[i] == y_policy_true[i])
    top3_hits.append(in_top_k(policy_probs[i], y_policy_true[i], 3))
    top5_hits.append(in_top_k(policy_probs[i], y_policy_true[i], 5))

    board = chess.Board(row["fen"])
    move = label_encoder.inverse_transform([policy_top1[i]])[0]
    try:
        board.push_san(move)
    except:
        illegal_moves += 1

illegal_rate = illegal_moves / len(test_df)

# ----------------------------------------------------
# 7. Value Head Error Analysis
# ----------------------------------------------------
value_errors = value_preds - y_value_true
abs_errors = np.abs(value_errors)

# Error buckets
buckets = {
    "Excellent (<0.05)": np.mean(abs_errors < 0.05),
    "Good (0.05–0.15)": np.mean((abs_errors >= 0.05) & (abs_errors < 0.15)),
    "Poor (0.15–0.3)": np.mean((abs_errors >= 0.15) & (abs_errors < 0.3)),
    "Very Poor (>0.3)": np.mean(abs_errors >= 0.3),
}

# ----------------------------------------------------
# 8. Print Final Results
# ----------------------------------------------------
print("\n========== POLICY HEAD RESULTS ==========")
print(f"Top-1 Accuracy     : {np.mean(top1_hits):.4f}")
print(f"Top-3 Accuracy     : {np.mean(top3_hits):.4f}")
print(f"Top-5 Accuracy     : {np.mean(top5_hits):.4f}")
print(f"Illegal Move Rate  : {illegal_rate * 100:.2f}%")

print("\n========== VALUE HEAD RESULTS ==========")
print(f"MAE                : {mean_absolute_error(y_value_true, value_preds):.4f}")
print(f"Median Abs Error   : {np.median(abs_errors):.4f}")
print(f"Max Abs Error      : {abs_errors.max():.4f}")

print("\n========== VALUE ERROR DISTRIBUTION ==========")
for k, v in buckets.items():
    print(f"{k:<22}: {v * 100:.2f}%")

# ----------------------------------------------------
# 9. Worst predictions (debugging insight)
# ----------------------------------------------------
analysis_df = test_df.copy()
analysis_df["pred_value"] = value_preds
analysis_df["abs_error"] = abs_errors
analysis_df["top1_correct"] = top1_hits
analysis_df["top3_correct"] = top3_hits
analysis_df["top5_correct"] = top5_hits

print("\n========== WORST VALUE PREDICTIONS ==========")
print(
    analysis_df
    .sort_values("abs_error", ascending=False)
    .head(5)[["fen", "value", "pred_value", "abs_error"]]
)

# ----------------------------------------------------
# 10. Save full report
# ----------------------------------------------------
analysis_df.to_csv("chess_model_result_analysis.csv", index=False)
print("\n✅ Detailed report saved: chess_model_result_analysis.csv")