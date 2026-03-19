from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import logging
import asyncio

# Local imports
from app.core.config import settings
from app.db import database
from app.services import binance_service, scanner_service, futures_service
from app.utils import market_utils as market_service_utils
from app.core.middleware import EndpointAuditMiddleware
from app.routes import spot, market, lead

# Configure logging for APScheduler
logging.basicConfig(level=logging.INFO)
logging.getLogger('apscheduler').setLevel(logging.INFO)

app = FastAPI(title="Trading Management API")
scheduler = AsyncIOScheduler()

# Setup Audit Middleware
app.add_middleware(EndpointAuditMiddleware)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(spot.router)
app.include_router(market.router)
app.include_router(lead.router, prefix="/lead")

# --- Scheduler Jobs ---
async def scheduled_scan_job():
    """Background job to run the scanner for a predefined set of pairs."""
    print("Running scheduled scan job...")
    try:
        # Dynamically find the top 20 opportunity pairs
        top_pairs = await asyncio.to_thread(market_service_utils.get_top_opportunity_pairs, 20)
        await asyncio.to_thread(scanner_service.run_scan, top_pairs)
        print("Scheduled scan completed.")
    except Exception as e:
        print(f"Error during scheduled scan: {e}")

async def scheduled_reconcile_job():
    """Background job to detect auto-closed trades."""
    print("Checking for auto-closed trades...")
    try:
        await asyncio.to_thread(binance_service.reconcile_trades)
    except Exception as e:
        print(f"Error during reconciliation: {e}")

# --- FastAPI Lifespan Events ---
@app.on_event("startup")
async def startup_event():
    # 1. Sync Binance Time
    try:
        binance_service.sync_binance_time()
    except: pass
    
    # 2. Database connectivity check
    database.ping_db()
    
    # 3. Start Scheduler
    scheduler.start()
    scheduler.add_job(
        scheduled_scan_job, 
        IntervalTrigger(minutes=settings.SCANNER_INTERVAL_MINUTES),
        id='scheduled_scanner',
        replace_existing=True
    )
    # Reconcile every 30 seconds
    scheduler.add_job(
        scheduled_reconcile_job,
        IntervalTrigger(seconds=30),
        id='scheduled_reconciler',
        replace_existing=True
    )
    print(f"Scheduler started. Reconciler runs every 30 seconds.")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    print("Scheduler shut down.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
