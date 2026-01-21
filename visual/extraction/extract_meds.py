import re
import logging
from typing import List, Dict, Optional
from PIL import Image
import pytesseract
import pandas as pd

# Import the matcher (assuming it's in the python path or relative)
# Adjust python path if needed in execution
import os

# Configure Tesseract Path (Before anything else)
if os.path.exists("/usr/users/infonum_projet_17/brito_fel/.conda/envs/ordonnances/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/usr/users/infonum_projet_17/brito_fel/.conda/envs/ordonnances/bin/tesseract"
    # Set explicit data path
    os.environ["TESSDATA_PREFIX"] = "/usr/users/infonum_projet_17/brito_fel/.conda/envs/ordonnances/share/tessdata"

# Import the matcher
try:
    from visual.matching.thesorimed_matcher import ThesorimedMatcher
except ImportError:
    # Fallback/Mock for standalone testing if path isn't set
    class ThesorimedMatcher:
        def __init__(self, db_path): pass
        def match(self, query): return [{"name": query, "score": 0.0}]

# Basic Regex for Posology
REGEX_PATTERNS = {
    # 500mg, 1g, 50 mg
    "strength": r"(\d+(?:[\.,]\d+)?)\s*(mg|g|ml|cp|gelule|sachet)",
    # 3 fois par jour, 3/j, 1 matin
    "frequency": r"(\d+|un|deux|trois)\s*(?:fois|x|prise)?\s*(?:par|/)\s*(?:jour|j|semaine)|(?:matin|midi|soir)",
    # pendant 5 jours, 1 mois
    "duration": r"(?:pendant|durant)?\s*(\d+)\s*(?:jour|j|semaine|mois|an)"
}

logger = logging.getLogger(__name__)

class MedicationExtractor:
    def __init__(self, use_matcher=True, db_path=None):
        self.matcher = None
        if use_matcher and db_path:
            try:
                self.matcher = ThesorimedMatcher(db_path)
                logger.info("Thesorimed Matcher loaded.")
            except Exception as e:
                logger.warning(f"Could not load Matcher: {e}")

    def ocr_crop(self, image: Image.Image) -> str:
        """
        Runs Tesseract OCR on a specific image crop.
        """
        # Configuration for French text, assuming single block
        custom_config = r'--oem 3 --psm 6 -l fra'
        text = pytesseract.image_to_string(image, config=custom_config)
        return text

    def parse_line(self, text: str) -> Dict:
        """
        Parses a single line of text into structural components.
        """
        result = {
            "raw_text": text.strip(),
            "drug": None,
            "strength": None,
            "frequency": None,
            "duration": None
        }
        
        # 1. Extract Strength
        m_str = re.search(REGEX_PATTERNS["strength"], text, re.IGNORECASE)
        if m_str:
            result["strength"] = m_str.group(0)
            # Remove strength from text to avoid confusion, or keep it? 
            # Keeping it for now, but simple extraction strategies usually consume tokens.
        
        # 2. Extract Frequency
        m_freq = re.search(REGEX_PATTERNS["frequency"], text, re.IGNORECASE)
        if m_freq:
            result["frequency"] = m_freq.group(0)

        # 3. Extract Duration
        m_dur = re.search(REGEX_PATTERNS["duration"], text, re.IGNORECASE)
        if m_dur:
            result["duration"] = m_dur.group(0)

        # 4. Extract Drug Name (Heuristic: The remaining text at the start)
        # Or better: Use the matcher to find the best substring
        if self.matcher:
            # Try to match the whole line or parts
            # Simple approach: Match the first 3-4 words
            words = text.split()
            candidate = " ".join(words[:4]) # Heuristic
            matches = self.matcher.match(candidate, top_k=1)
            if matches and matches[0]['score'] > 60:
                result["drug"] = matches[0]['name']
            else:
                result["drug"] = candidate # Fallback
        else:
            result["drug"] = text.split()[0] if text else "Unknown"

        return result

    def extract_from_crop(self, image: Image.Image) -> List[Dict]:
        """
        Full pipeline: OCR -> Line Splitting -> Parsing
        """
        full_text = self.ocr_crop(image)
        lines = [l.strip() for l in full_text.split('\n') if l.strip()]
        
        extracted_meds = []
        for line in lines:
            # Filter noise lines (too short)
            if len(line) < 5: continue
            
            med_data = self.parse_line(line)
            extracted_meds.append(med_data)
            
        return extracted_meds

if __name__ == "__main__":
    # Test Block
    print("Testing Extractor...")
    # Mock image or load one if available
    # ex = MedicationExtractor(db_path="thesorimed.db")
    pass
