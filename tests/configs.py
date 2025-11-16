from pathlib import Path

TEST_DIR = Path(__file__).parent.absolute()
GENERATED_DIR = TEST_DIR / "generated"

GENERATED_DIR.mkdir(parents=True, exist_ok=True)
