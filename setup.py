"""Setup script for development environment."""

#!/usr/bin/env python3
"""Setup script for physics simulation project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up Physics Simulation with GNNs project...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies"),
        ("pip install -e .", "Installing package in development mode"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"Setup failed at: {description}")
            sys.exit(1)
    
    # Install pre-commit hooks
    if run_command("pre-commit install", "Installing pre-commit hooks"):
        print("✓ Pre-commit hooks installed")
    else:
        print("⚠ Pre-commit hooks installation failed (optional)")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "logs",
        "assets/plots",
        "assets/models",
        "assets/embeddings"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create .gitkeep files
    gitkeep_dirs = ["data/raw", "data/processed"]
    for directory in gitkeep_dirs:
        gitkeep_file = Path(directory) / ".gitkeep"
        gitkeep_file.touch()
        print(f"✓ Created: {gitkeep_file}")
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run tests: pytest tests/")
    print("2. Launch demo: streamlit run demo/streamlit_app.py")
    print("3. Train model: python scripts/train.py")
    print("4. Run main script: python main.py --help")


if __name__ == "__main__":
    main()
