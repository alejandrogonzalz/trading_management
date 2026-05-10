# Operational Protocols & Constraints

## Git — Windows/PowerShell (CRITICAL)
`&&` operator fails in PowerShell 5.1. **Never chain git commands with `&&`.**
Always run git sequentially in separate tool calls or use `;` (unconditional separator).

```powershell
# WRONG
git add . && git commit -m "msg"

# CORRECT
git add .
git commit -m "msg"
# or
git add .; git commit -m "msg"
```

## Python Environment
- LangGraph/backtest/optimization runs in conda env `trading` (Python 3.11)
- Backend runs inside Docker (`docker-compose exec backend bash`)
- Never assume system Python — always specify conda env context when running scripts

```powershell
conda activate trading
python langgraph/cli.py run-backtest ...
```

## Smart Trade Safety — Non-Negotiable Rules
- **Floor Rounding**: Applied to ALL quantities without exception
- **ID-Strict Reconciliation**: Never delete trades while Binance orders are active
- **Fee-Aware Clipping**: Commissions auto-deducted from sell quantities (prevents "Insufficient Balance")
- **Time Sync**: `recvWindow=60000` + `binance_service.sync_time()` on API timeout
- **KNOWN LIMITATION**: If app offline for extended period, Lead reconciler uses current ticker price as `exit_price` instead of historical fill — fix pending (scan historical fills)

## Database Rules
- SQLite WAL mode — never disable
- All modifications go through SQLAlchemy ORM; no raw SQL writes in business logic
- Audit trail (`audit_log`) must capture all API interactions
- Backup DB before any destructive script: `backend/scripts/db_manager.py`

## Docker Services
```powershell
docker-compose up -d --build        # Start all services
docker-compose logs backend -f      # Stream backend logs
docker-compose exec backend bash    # Enter backend container
```
Ports: backend=8001, frontend=5173, langgraph=2024, ollama=11434

## Testing
- pytest for backend unit/integration tests
- Tests in `backend/app/tests/` and `backend/tests/`
- Always run tests before committing changes to reconciliation or order-placement logic

## Commit Style
Short imperative subject line. No co-author tags unless explicitly requested.
