import os
import json
import asyncio
import time
from typing import Optional
from abc import ABC, abstractmethod
from langchain_core.messages import SystemMessage, HumanMessage


class LLMProvider(ABC):
    """Thin wrapper for LLM interactions."""

    @abstractmethod
    async def generate_setup(self, system_prompt: str, user_prompt: str) -> str:
        """Sends messages to LLM and returns content string."""
        pass


class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str, base_url: str, temperature: float):
        from langchain_ollama import ChatOllama

        self.client = ChatOllama(
            model=model_name, base_url=base_url, temperature=temperature
        )

    async def generate_setup(self, system_prompt: str, user_prompt: str) -> str:
        max_retries = 3
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = await self.client.ainvoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )

                content = (
                    response.content
                    if response and hasattr(response, "content")
                    else ""
                )

                # Validate response is not empty
                if not content or content.strip() == "":
                    raise ValueError("Empty response from Ollama")

                # Basic validation - should contain JSON structure
                if "{" not in content and "[" not in content:
                    # Might still be valid if it's a simple value, but warn
                    print(
                        f"Warning: LLM response may not be JSON (attempt {attempt + 1}/{max_retries})"
                    )

                return content

            except Exception as e:
                last_exception = e
                print(
                    f"Ollama API error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                )

                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1, 2, 4 seconds
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue

        # All retries failed
        error_msg = f"Failed after {max_retries} attempts: {last_exception}"
        print(error_msg)

        # Return a minimal valid JSON structure as fallback
        fallback_response = {
            "error": error_msg,
            "bias": "NEUTRAL",
            "entry": 0,
            "tp": 0,
            "sl": 0,
            "leverage": None,
            "reasoning": f"LLM service unavailable: {last_exception}",
        }
        return json.dumps(fallback_response)


class GoogleProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: str, temperature: float):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.client = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key, temperature=temperature
        )

    async def generate_setup(self, system_prompt: str, user_prompt: str) -> str:
        max_retries = 3
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = await self.client.ainvoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )

                content = (
                    response.content
                    if response and hasattr(response, "content")
                    else ""
                )

                # Validate response is not empty
                if not content or content.strip() == "":
                    raise ValueError("Empty response from Google Generative AI")

                # Basic validation - should contain JSON structure
                if "{" not in content and "[" not in content:
                    print(
                        f"Warning: LLM response may not be JSON (attempt {attempt + 1}/{max_retries})"
                    )

                return content

            except Exception as e:
                last_exception = e
                print(
                    f"Google AI API error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                )

                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    continue

        # All retries failed
        error_msg = f"Failed after {max_retries} attempts: {last_exception}"
        print(error_msg)

        # Return a minimal valid JSON structure as fallback
        fallback_response = {
            "error": error_msg,
            "bias": "NEUTRAL",
            "entry": 0,
            "tp": 0,
            "sl": 0,
            "leverage": None,
            "reasoning": f"LLM service unavailable: {last_exception}",
        }
        return json.dumps(fallback_response)


class BedrockProvider(LLMProvider):
    def __init__(self, model_name: str, region: str, temperature: float):
        from langchain_aws import ChatBedrock

        self.client = ChatBedrock(
            model_id=model_name,
            model_kwargs={"temperature": temperature},
            region_name=region,
        )

    async def generate_setup(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return response.content


class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str, temperature: float):
        from langchain_openai import ChatOpenAI

        self.client = ChatOpenAI(model=model_name, temperature=temperature)

    async def generate_setup(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return response.content


class MockProvider(LLMProvider):
    async def generate_setup(self, system_prompt: str, user_prompt: str) -> str:
        """Returns mock trade setups based on the symbol and scenario indicators."""
        # Detect if we are in optimizer mode
        is_optimizer = (
            "optimizer" in system_prompt.lower() or "Risk Manager" in system_prompt
        )

        # Simple detection of symbol and scenario from prompt
        if "BTCUSDT" in user_prompt:
            if "STRONG_BEARISH" in user_prompt:
                setup = {
                    "bias": "SHORT",
                    "entry": 61900.0,
                    "tp": 58000.0,
                    "sl": 63000.0,
                    "leverage": 10,
                    "reasoning": "Strong bearish trend on multiple timeframes.",
                    "quality": "HIGH",
                    "confidence": 8,
                }
            else:  # strong_bullish
                setup = {
                    "bias": "LONG",
                    "entry": 65100.0,
                    "tp": 68000.0,
                    "sl": 64000.0,
                    "leverage": 10,
                    "reasoning": "Strong bullish trend and breakout detected.",
                    "quality": "HIGH",
                    "confidence": 9,
                }
        elif "LINKUSDT" in user_prompt:  # high_volatility_futures
            setup = {
                "bias": "LONG",
                "entry": 18.6,
                "tp": 22.0,
                "sl": 18.0,
                "leverage": 5,
                "reasoning": "Breakout with high volume, adjusting leverage for volatility.",
                "quality": "MEDIUM",
                "confidence": 7,
            }
        elif "ETHUSDT" in user_prompt:  # overbought_spot
            setup = {
                "bias": "LONG",
                "entry": 3510.0,
                "tp": 4000.0,
                "sl": 3400.0,
                "leverage": None,
                "reasoning": "Bullish structure despite overbought RSI.",
                "quality": "MEDIUM",
                "confidence": 6,
            }
        elif "SOLUSDT" in user_prompt:
            if (
                "RANGE" in user_prompt or "NEUTRAL" in user_prompt
            ):  # range_market or risky_futures
                setup = {
                    "bias": "LONG",
                    "entry": 146.0,
                    "tp": 155.0,
                    "sl": 140.0,
                    "leverage": 25,  # High leverage to trigger risk issue for optimizer
                    "reasoning": "Low volatility range, playing the bounce.",
                    "quality": "LOW",
                    "confidence": 4,
                }
            else:
                setup = {
                    "bias": "LONG",
                    "entry": 146.0,
                    "tp": 160.0,
                    "sl": 142.0,
                    "leverage": 10,
                    "reasoning": "Bullish momentum building.",
                    "quality": "MEDIUM",
                    "confidence": 7,
                }
        else:
            # Try to extract indicators from user_prompt to generate realistic mock
            price = 100.0
            volatility_multiplier = 1.0
            heatmap = "NEUTRAL"
            try:
                # Find the JSON part after "Indicators: "
                import re

                match = re.search(r"Indicators:\s*(\{.*\})", user_prompt, re.DOTALL)
                if match:
                    indicators_json = match.group(1)
                    indicators = json.loads(indicators_json)
                    # Get first timeframe's data
                    first_tf = next(iter(indicators.values())) if indicators else {}
                    # Try to find price in various fields
                    price = (
                        first_tf.get("close")
                        or first_tf.get("price")
                        or first_tf.get("last")
                        or 100.0
                    )
                    if price <= 0:
                        price = 100.0
                    # Adjust for volatility
                    atr_ratio = first_tf.get("atr_ratio", 1.0)
                    volatility_multiplier = max(0.5, min(atr_ratio, 3.0))
                    # Get heatmap for bias decision
                    heatmap = first_tf.get("heatmap", "NEUTRAL")
            except Exception:
                pass  # Keep default values

            # Determine if SPOT mode
            is_spot = "SPOT" in system_prompt and "FUTURES" not in system_prompt

            # Determine bias based on mode and heatmap
            if is_spot:
                bias = "LONG"  # SPOT only supports LONG
            else:
                # FUTURES: decide based on heatmap
                if "BEARISH" in heatmap.upper():
                    bias = "SHORT"
                else:
                    bias = "LONG"

            # Generate realistic setup based on price
            entry = round(price, 2)
            # TP/SL percentages based on mode and volatility
            if is_spot:
                tp_pct = 0.05 * volatility_multiplier  # 5% target
                sl_pct = 0.03 * volatility_multiplier  # 3% stop
                leverage = None
            else:
                tp_pct = 0.03 * volatility_multiplier  # 3% target
                sl_pct = 0.02 * volatility_multiplier  # 2% stop
                leverage = 5  # Default leverage

            # Adjust TP/SL direction based on bias
            if bias == "LONG":
                tp = round(entry * (1 + tp_pct), 2)
                sl = round(entry * (1 - sl_pct), 2)
            else:  # SHORT
                tp = round(entry * (1 - tp_pct), 2)
                sl = round(entry * (1 + sl_pct), 2)

            setup = {
                "bias": bias,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "leverage": leverage,
                "reasoning": f"Mock setup based on price ${price:.2f}, volatility, and {heatmap.lower()} bias.",
                "quality": "MEDIUM",
                "confidence": 6,
            }

        if is_optimizer:
            setup["changes"] = ["Refined entry and tightened SL for better RR."]
            setup["reasoning"] = "Optimized setup to resolve identified issues."

        return json.dumps(setup)


def get_llm_provider() -> LLMProvider:
    """Factory to create the configured LLM provider."""
    provider_type = os.getenv("LLM_PROVIDER", "ollama").lower()
    model_name = os.getenv("LLM_MODEL", "qwen2.5:14b")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    if provider_type == "mock":
        return MockProvider()

    if provider_type == "google":
        return GoogleProvider(
            model_name=model_name,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )

    elif provider_type == "bedrock":
        return BedrockProvider(
            model_name=model_name,
            region=os.getenv("AWS_REGION", "us-east-1"),
            temperature=temperature,
        )

    elif provider_type == "openai":
        return OpenAIProvider(model_name=model_name, temperature=temperature)

    else:  # Default to Ollama
        return OllamaProvider(
            model_name=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
            temperature=temperature,
        )
