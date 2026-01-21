import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class BenchmarkTracker:
    def __init__(self, history_file: str = "history.json"):
        self.history_path = Path(history_file)
        # Ensure directory exists but be careful not to create a dir named history.json if user passed a file path
        if not self.history_path.parent.exists():
             self.history_path.parent.mkdir(parents=True, exist_ok=True)
             
        self.history = self._load_history()

    def _load_history(self) -> list:
        if self.history_path.exists():
            try:
                with open(self.history_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode {self.history_path}, starting fresh.")
                return []
        return []

    def log_run(self, 
                model_name: str, 
                metrics: Dict[str, float], 
                config: Optional[Dict[str, Any]] = None,
                tags: Optional[list] = None):
        """
        Log a new benchmark run.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "metrics": metrics,
            "config": config or {},
            "tags": tags or []
        }
        
        self.history.append(entry)
        self._save_history()
        logger.info(f"Logged benchmark for {model_name}")

    def _save_history(self):
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def print_summary(self):
        """
        Print a table of recent runs.
        """
        if not self.history:
            print("No history available.")
            return

        print("\n" + "="*80)
        print(f"{'TIMESTAMP':<25} | {'MODEL':<25} | {'IoU HEADER':<10} | {'IoU MEDS':<10}")
        print("-" * 80)
        
        # Show last 10 entries
        for entry in self.history[-10:]:
            ts = entry['timestamp'][:19]
            mod = entry['model_name'][:25]
            iou_h = entry['metrics'].get('iou_header', 0.0)
            iou_m = entry['metrics'].get('iou_med_block', 0.0)
            
            print(f"{ts:<25} | {mod:<25} | {iou_h:<10.4f} | {iou_m:<10.4f}")
        print("="*80 + "\n")

if __name__ == "__main__":
    # Test
    tracker = BenchmarkTracker("./history_test.json")
    tracker.log_run("test_model_v1", {"iou_header": 0.5, "iou_med_block": 0.6})
    tracker.print_summary()
    # Cleanup test
    if Path("./history_test.json").exists():
        os.remove("./history_test.json")
