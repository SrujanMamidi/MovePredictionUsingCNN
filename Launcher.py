import subprocess
import sys
import gradio as gr


def start_chess():
    subprocess.Popen(
        [sys.executable, "app.py"],
        cwd="move prediction in turn based games"
    )
    return "Chess started. Check the new tab/window."


def start_go():
    subprocess.Popen(
        [sys.executable, "main.py"],
        cwd="Little-Go-Game-main"
    )
    return "Go game started. Check the new tab/window."


css = """
.gradio-container {
    background:
        radial-gradient(circle at 15% 15%, rgba(255, 215, 120, 0.22), transparent 28%),
        radial-gradient(circle at 88% 12%, rgba(130, 180, 255, 0.25), transparent 34%),
        linear-gradient(140deg, #ecf2fb 0%, #e5efe8 48%, #f9f2e7 100%);
    min-height: 100vh;
    padding-top: 12px;
}

#title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    color: #1a2435;
    letter-spacing: 0.4px;
    margin-bottom: 8px;
}

#subtitle {
    text-align: center;
    color: #3f4d63;
    margin-bottom: 24px;
    font-size: 16px;
}

.game-card {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(44, 61, 88, 0.15);
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 16px 34px rgba(24, 35, 54, 0.15);
    min-height: 220px;
}

.game-title {
    font-size: 30px;
    font-weight: 800;
    color: #1f2d44;
    margin-bottom: 6px;
}

.game-desc {
    color: #4a5a74;
    margin-bottom: 16px;
    font-size: 15px;
}

#chess-btn button,
#go-btn button {
    width: 100%;
    font-size: 18px !important;
    font-weight: 700 !important;
    padding: 14px 16px !important;
    border-radius: 12px !important;
    border: 0 !important;
}

#chess-btn button {
    background: linear-gradient(135deg, #3459d6, #5f8ff0) !important;
    color: #ffffff !important;
}

#go-btn button {
    background: linear-gradient(135deg, #1f9c8b, #46c3ad) !important;
    color: #ffffff !important;
}

#status-box textarea {
    background: #f4f8ff !important;
    border: 1px solid #cfddf3 !important;
    border-radius: 10px !important;
    color: #1f2c40 !important;
    font-weight: 600 !important;
}

#status-box label {
    color: #293954 !important;
    font-weight: 700 !important;
}
"""


with gr.Blocks(css=css, title="Game Selector") as demo:
    gr.Markdown("<div id='title'> Game Center</div>")
    gr.Markdown("<div id='subtitle'>Choose a game and launch it in a new window.</div>")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="game-card"):
            gr.Markdown("<div class='game-title'>Chess</div>")
            gr.Markdown("<div class='game-desc'>Play chess with move prediction.</div>")
            chess_btn = gr.Button("Start Chess", variant="primary", elem_id="chess-btn")

        with gr.Column(scale=1, elem_classes="game-card"):
            gr.Markdown("<div class='game-title'>Go</div>")
            gr.Markdown("<div class='game-desc'>Play the classic strategy board game.</div>")
            go_btn = gr.Button("Start Go", variant="primary", elem_id="go-btn")

    gr.Markdown("")

    status = gr.Textbox(
        label="Status",
        interactive=False,
        placeholder="Game status will appear here...",
        elem_id="status-box",
    )

    chess_btn.click(start_chess, outputs=status)
    go_btn.click(start_go, outputs=status)


demo.launch()
