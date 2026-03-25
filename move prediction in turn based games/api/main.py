from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chess
import numpy as np
import tensorflow as tf
import joblib
from utils.encoding import encode_board
import os

app = FastAPI(title="Chess Move Prediction API")

# --- Load Model and Encoder ---
MODEL_PATH = 'models/best_model.h5'
ENCODER_PATH = 'label_encoder.joblib'

model = None
encoder = None

@app.on_event("startup")
def load_assets():
    global model, encoder
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        print(f"Warning: Could not load model or encoder. API will fail on /predict. Error: {e}")

class PredictRequest(BaseModel):
    fen: str

class MovePrediction(BaseModel):
    move: str
    probability: float

class PredictResponse(BaseModel):
    best_move: str
    top_3_moves: list[MovePrediction]
    evaluation: float
    side_to_move: str

@app.post("/predict", response_model=PredictResponse)
def predict_move(request: PredictRequest):
    if model is None or encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        board = chess.Board(request.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")
        
    encoded = np.expand_dims(encode_board(board), axis=0)
    policy_probs, value_score = model.predict(encoded, verbose=0)
    
    # Get Top 3 moves
    probs = policy_probs[0]
    top_indices = np.argsort(probs)[-3:][::-1]
    
    top_3 = []
    for idx in top_indices:
        move_san = encoder.inverse_transform([idx])[0]
        top_3.append(MovePrediction(move=move_san, probability=float(probs[idx])))
        
    best_move = top_3[0].move
    evaluation = float(value_score[0][0])
    side = "White" if board.turn == chess.WHITE else "Black"
    
    return PredictResponse(
        best_move=best_move,
        top_3_moves=top_3,
        evaluation=evaluation,
        side_to_move=side
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
