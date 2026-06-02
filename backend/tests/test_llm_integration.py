import asyncio
import os
import sys
from unittest.mock import AsyncMock, Mock, patch

# Ensure app can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services import llm_service


async def test_langgraph_transformation():
    print("--- TESTING LLM SERVICE TRANSFORMATION ---")

    # Mock the HTTP response from LangGraph
    mock_langgraph_response = {
        "run_id": "test-run-123",
        "symbol": "BTCUSDT",
        "mode": "FUTURES",
        "original": {
            "bias": "LONG",
            "entry": 65000.0,
            "tp": 68000.0,
            "sl": 64000.0,
            "leverage": 10,
            "reasoning": "Strong trend.",
        },
        "evaluation": {
            "confidence": 9,
            "quant_confidence": 8.5,
            "rr": 3.0,
            "safety_margin": 0.05,
            "liquidation_safe": True,
        },
        "issues": [],
        "audit_trail": [],
    }

    # Patch httpx.AsyncClient to return our mock
    with patch("httpx.AsyncClient") as mock_client_cls:
        # Create the mock client instance
        mock_client_instance = AsyncMock()

        # Setup the context manager to return the instance
        mock_client_instance.__aenter__.return_value = mock_client_instance

        # Setup the response object (Standard Mock for attributes, not AsyncMock)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_langgraph_response

        # Setup post to be awaitable and return our response
        mock_client_instance.post.return_value = mock_response

        # Assign the class to return our instance
        mock_client_cls.return_value = mock_client_instance

        # Call the service function
        result = await llm_service.get_deep_langgraph_analysis("BTCUSDT", {}, "FUTURES")

        print("\n[TRANSFORMED RESULT]")
        import pprint

        pprint.pprint(result)

        # ASSERTIONS
        print("\n[VALIDATION]")
        try:
            assert result["pair"] == "BTCUSDT"
            assert result["entry"] == 65000.0
            assert result["tp"] == 68000.0
            assert result["sl"] == 64000.0
            assert result["leverage"] == 10
            assert result["risk_reward"] == "1:3.0"
            assert result["confidence"] == 9
            print("✅ All fields match Frontend expectations.")
        except AssertionError as e:
            print(f"❌ Assertion Failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_langgraph_transformation())
