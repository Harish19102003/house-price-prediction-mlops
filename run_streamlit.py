#!/usr/bin/env python3
"""
Script to run the Streamlit house price prediction app.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit app."""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Path to the Streamlit app
    app_path = project_root / "src" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Streamlit app not found at: {app_path}")
        sys.exit(1)
    
    print("🚀 Starting House Price Predictor Streamlit App...")
    print(f"📁 App location: {app_path}")
    print("🌐 The app will open in your default browser")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 