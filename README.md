# 🎮 Move Prediction Using CNN (Chess & Go AI)

Welcome to the **Move Prediction in Turn-Based Games** repository! This project combines advanced Deep Learning techniques and classic AI search algorithms to evaluate board states and predict optimal moves in popular turn-based strategy games: **Chess** and **Little-Go (5x5)**.

The project features a unified, interactive **Gradio Launcher** that allows users to easily navigate and execute either game simulation from a single interface.

---

## 🌟 Key Features

1. **Unified Game Launcher** 🎯
   - A beautiful Gradio-based web interface (`Launcher.py`) that acts as a central hub to launch either the Chess or Go AI environments with a single click.

2. **Chess Move Prediction (CNN / ResNet)** ♟️
   - Implements a state-of-the-art RestNet-style Convolutional Neural Network (CNN) to predict the next best chess move.
   - Evaluates the board via FEN string input.
   - Includes a dual-headed network for both **Policy (Best Move)** and **Value (Position Evaluation)**.
   - Features a modern Gradio UI (`app.py`) for real-time visualization of the predicted move on the board.

3. **Little-Go Game (5x5) & Tournament Simulator** ⚪⚫
   - A complete Tkinter-based GUI framework simulating 5x5 Go.
   - Includes multiple AI bots: Random, Greedy, Aggressive, and a custom **Minimax with Alpha-Beta Pruning** agent.
   - Automated tournament mode (`tournament.py`) to benchmark different AI strategies against each other.

---

## 🏗️ Repository Structure

```text
📦 MovePredictionUsingCNN
 ┣ 📂 move prediction in turn based games/  # ♟️ Chess CNN Prediction System
 ┃ ┣ 📂 models/                             # ResNet model architecture
 ┃ ┣ 📂 api/                                # FastAPI backend
 ┃ ┣ 📂 ui/                                 # Gradio UI components
 ┃ ┣ 📂 training/                           # Training pipelines & data augmentation
 ┃ ┣ 📜 app.py                              # Entry point for the Chess interface
 ┃ ┗ ...
 ┣ 📂 Little-Go-Game-main/                  # ⚪⚫ 5x5 Go Game Simulation
 ┃ ┣ 📂 players/                            # AI Strategy scripts (Minimax, Greedy, etc.)
 ┃ ┣ 📜 main.py                             # Entry point for the Go Tkinter GUI
 ┃ ┣ 📜 tournament.py                       # Automated bot tournament simulator
 ┃ ┗ ...
 ┗ 📜 Launcher.py                           # 🚀 Main Unified Gradio Hub
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Installation
Clone the repository and install the necessary dependencies:

```bash
# Clone the repository
git clone https://github.com/SrujanMamidi/MovePredictionUsingCNN.git
cd MovePredictionUsingCNN

# (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages (Make sure to install sub-project requirements if any)
pip install gradio chess tensorflow python-chess
# Or if pip requirements are provided inside the chess subfolder:
pip install -r "move prediction in turn based games/requirements.txt"
```

### 3. Running the unified Launcher
To start the Gradio hub and choose which game you want to launch:

```bash
python Launcher.py
```
This will open a beautiful web interface. Clicking on **"Start Chess"** or **"Start Go"** will directly launch their respective AI applications in a new window!

---

## 🤖 Game-Specific Details

### ♟️ Chess AI
- **Architecture**: A deep Residual Neural Network (ResNet).
- **Input**: 8x8x17 structured tensor mapping the board state, castling rights, and active turn.
- **Usage**: You can also run the chess predictor independently by navigating to its folder and running `python app.py`.

### ⚪⚫ Little-Go AI
- **Algorithms Used**: Minimax with Alpha-Beta Pruning, Greedy heuristics, and Multi-ply lookaheads.
- **Simulation**: Play interactively against an AI, or pit two AI bots against each other.
- **Usage**: You can run the Go GUI independently by navigating to its folder and running `python main.py`.

---

*This repository demonstrates the power of integrating Neural Networks for complex state spaces (Chess) alongside traditional Search Algorithms (Go) within a single ecosystem.*
