import subprocess
import sys
import os

if __name__ == "__main__":
    port = os.environ.get("PORT", "8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])