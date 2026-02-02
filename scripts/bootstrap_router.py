"""
Bootstrap Script for SBSCR Router.
1. Runs Calibration (if data exists)
2. Trains the XGBoost Student
3. Maps signatures for Fast-Path
"""

import sys
import os
import subprocess

# Ensure path is set
sys.path.append(os.getcwd())

from sbscr.routers.sbscr import SBSCRRouter

def main():
    print("🛠️  Bootstraping SBSCR Router...")
    
    # 1. Calibrate (Optional Step)
    # Check if we should call gemini_calibrate.py
    # For this demo, we assume the user might want to run it manually or we start with bootstrap
    
    if not os.path.exists("data/calibrated_dataset.json"):
        print("💡 Step 1: No calibrated dataset found. RUNNING CALIBRATION (Partial)...")
        subprocess.run([sys.executable, "scripts/gemini_calibrate.py"])
    
    # 2. Train XGBoost
    print("\n💡 Step 2: Training XGBoost Scorer (The Student)...")
    subprocess.run([sys.executable, "scripts/train_enterprise.py"])
    
    # 3. Signature Mapping
    print("\n💡 Step 3: Mapping LSH Signatures (The Fast Path)...")
    router = SBSCRRouter()
    router.teach_from_dataset("data/calibrated_dataset.json")
    
    print("\n✅ Router Bootstrap Complete! Ready for Deployment.")

if __name__ == "__main__":
    main()
