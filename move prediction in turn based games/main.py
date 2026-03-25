import subprocess
import time
import sys
import os

def main():
    print("🚀 Starting Chess Move Prediction System...")
    
    # Check if model exists
    if not os.path.exists("models/best_model.h5"):
        print("❌ Error: 'models/best_model.h5' not found.")
        print("👉 You must train the model first by running: python -m training.train")
        return

    # 1. Start Fast API in background
    print("📡 Starting API Server...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.getcwd()
    )
    
    # Wait for API to start
    time.sleep(3)
    
    # 2. Start UI
    print("🖥️ Starting UI Client...")
    try:
        subprocess.run(
            [sys.executable, "ui/app.py"],
            cwd=os.getcwd()
        )
    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        api_process.terminate()

if __name__ == "__main__":
    main()
