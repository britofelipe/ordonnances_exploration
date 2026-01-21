import sys
import os
from pathlib import Path
from PIL import Image

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from visual.extraction.extract_meds import MedicationExtractor

def test_extraction():
    # 1. Setup
    db_path = project_root / "data/thesorimed/THESORIMED_SQ3"
    img_path = project_root / "visual/dataset_v1/images/ordo_vis_000000.png"
    
    if not db_path.exists():
        print(f"Error: DB not found at {db_path}")
        return
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        return

    # 2. Init Extractor
    print("Initializing Extractor...")
    extractor = MedicationExtractor(db_path=str(db_path))
    
    # 3. Load Image
    img = Image.open(img_path)
    
    # 4. Run Extraction (Whole image test, assuming crop logic is external)
    print(f"Processing {img_path.name}...")
    results = extractor.extract_from_crop(img)
    
    # 5. Report
    print("\n--- Extracted Results ---")
    for i, res in enumerate(results, 1):
        print(f"Item {i}:")
        print(f"  Raw: {res['raw_text']}")
        print(f"  Drug: {res['drug']}")
        print(f"  Dose: {res['strength']}")
        print(f"  Freq: {res['frequency']}")
        print("-" * 20)

if __name__ == "__main__":
    test_extraction()
