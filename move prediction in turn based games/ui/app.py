import gradio as gr
import chess
import chess.svg
import requests
import json
import base64

# API configuration
API_URL = "http://localhost:8000/predict"

def get_board_svg(board, last_move=None):
    """Generates a nice SVG for the board."""
    fill = {}
    if last_move:
        fill[last_move.from_square] = "#ccffcc"
        fill[last_move.to_square] = "#ccffcc"
    
    return chess.svg.board(board=board, fill=fill, size=500)

def predict_move(fen):
    try:
        response = requests.post(API_URL, json={"fen": fen})
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def process_fen(fen):
    if not fen:
        fen = chess.STARTING_FEN
        
    try:
        board = chess.Board(fen)
    except:
        return "Invalid FEN", None, None, None, None

    # Get predictions
    data = predict_move(fen)
    if not data:
        return "API Error. Ensure backend is running.", get_board_svg(board), "Error", "Error", []

    best_move = data['best_move']
    eval_score = data['evaluation']
    top_3 = data['top_3_moves']
    
    # Try to parse the move for highlighting
    last_move = None
    try:
        last_move = board.parse_san(best_move)
    except:
        pass
        
    board_svg = get_board_svg(board, last_move)
    
    # Format top 3
    top_3_html = "<ul>"
    for m in top_3:
        top_3_html += f"<li><b>{m['move']}</b>: {m['probability']*100:.1f}%</li>"
    top_3_html += "</ul>"
    
    # Eval text
    eval_color = "green" if eval_score > 0 else "red"
    eval_html = f"<div style='font-size: 24px; color: {eval_color};'><b>{eval_score:+.2f}</b></div>"
    
    return best_move, board_svg, eval_html, f"Side: {data['side_to_move']}", top_3_html

# --- Custom UI ---
with gr.Blocks(title="Grandmaster AI") as demo:
    gr.Markdown("""
    # ♟️ Grandmaster AI: Move Prediction
    *State-of-the-Art Chess Move Prediction using Residual Neural Networks.*
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            board_display = gr.HTML(label="Chess Board")
            fen_input = gr.Textbox(label="FEN Position", placeholder="Enter FEN here...", value=chess.STARTING_FEN)
            btn = gr.Button("Analyze Position", variant="primary")
            
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🤖 Model Insights")
                best_move_out = gr.Label(label="Predicted Best Move")
                eval_out = gr.HTML(label="Position Evaluation")
                side_out = gr.Text(label="To Move")
                
            with gr.Group():
                gr.Markdown("### 📊 Top 3 Candidate Moves")
                top_3_out = gr.HTML()

    btn.click(
        process_fen, 
        inputs=[fen_input], 
        outputs=[best_move_out, board_display, eval_out, side_out, top_3_out]
    )
    
    # Initial load
    demo.load(process_fen, inputs=[fen_input], outputs=[best_move_out, board_display, eval_out, side_out, top_3_out])

if __name__ == "__main__":
    demo.launch(server_port=7860, theme=gr.themes.Soft())
