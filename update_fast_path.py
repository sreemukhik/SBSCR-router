from sbscr.routers.sbscr import SBSCRRouter
import os

def update_map():
    print("🎓 Mapping 612 Golden Instructions to the Fast-Path Cache...")
    router = SBSCRRouter()
    
    calibrated_path = "data/calibrated_dataset.json"
    if os.path.exists(calibrated_path):
        router.teach_from_dataset(calibrated_path)
        print("✅ Signature Map Updated Successfully!")
    else:
        print("❌ Calibrated dataset not found.")

if __name__ == "__main__":
    update_map()
