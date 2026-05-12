"""Configure sys.path so all test subfolders can import langgraph packages."""
import sys
from pathlib import Path

# Add langgraph/ root to path so `from backtest.x import y` and `from agent.x import y` work
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
