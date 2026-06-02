from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.middleware import EndpointAuditMiddleware
from app.db import database
from app.routes import lead, market, spot
from app.services import binance_service
from app.services.futures_service import futures_service

# Initialize FastAPI
app = FastAPI(title="Trading Management API")

# Add Audit Middleware
app.add_middleware(EndpointAuditMiddleware)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(spot.router, tags=["Spot"])
app.include_router(market.router, tags=["Market"])
app.include_router(lead.router, prefix="/lead", tags=["Lead"])

# Setup Scheduler
scheduler = AsyncIOScheduler()


@app.on_event("startup")
async def startup_event():
    # 0. Initialize SQLite
    database.init_db()

    # 1. Sync Binance Time
    try:
        binance_service.sync_binance_time()
    except:
        pass

    # 2. Database connectivity check
    database.ping_db()

    # 3. LLM Warm-up (Disabled for Cloud/Factory compatibility)
    # asyncio.create_task(llm_service.warm_up_llm())

    # 4. Schedule Reconcilers
    scheduler.add_job(
        binance_service.reconcile_trades,
        trigger=IntervalTrigger(seconds=30),
        id="scheduled_reconcile_job",
        replace_existing=True,
    )

    scheduler.add_job(
        futures_service.reconcile_lead_trades,
        trigger=IntervalTrigger(seconds=30),
        id="scheduled_lead_reconcile_job",
        replace_existing=True,
    )

    scheduler.start()
    print("Scheduler started. Reconcilers run every 30 seconds.")


@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    print("Scheduler shut down.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
