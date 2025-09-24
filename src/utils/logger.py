
import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "app.log"

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

handler = logging.FileHandler(LOG_PATH)
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setFormatter(formatter)

logger = logging.getLogger("sales_insights")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(handler)
    logger.addHandler(console)

# convenience
def get_logger(name: str = "sales_insights"):
    return logging.getLogger(name)
