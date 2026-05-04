import uvicorn
import uuid
from fastapi import FastAPI
from pydantic import BaseModel, Field
from agent.graph import graph
from typing import Dict, Any, Literal

app = FastAPI(title="Trading LangGraph Agent")


class AnalysisRequest(BaseModel):
    symbol: str
    mode: Literal["SPOT", "FUTURES"] = Field(default="SPOT")
    indicators: Dict[str, Any]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "langgraph"}


@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    # Unique Run ID for this execution
    run_id = str(uuid.uuid4())

    # Initial State for the TradeState TypedDict
    initial_state = {
        "run_id": run_id,
        "symbol": request.symbol,
        "mode": request.mode,
        "indicators": request.indicators,
        "original": None,
        "evaluation": None,
        "optimized": None,
        "issues": [],
        "needs_optimization": False,
        "audit_trail": [],  # Starts empty, nodes append to it
    }

    # Run the graph
    result = await graph.ainvoke(initial_state)

    # Return structured results including the audit trail
    return {
        "run_id": result["run_id"],
        "symbol": result["symbol"],
        "mode": result["mode"],
        "original": result.get("original"),
        "evaluation": result.get("evaluation"),
        "optimized": result.get("optimized"),
        "issues": result.get("issues", []),
        "audit_trail": result.get("audit_trail", []),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2024)
