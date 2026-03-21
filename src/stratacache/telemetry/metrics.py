from stratacache.telemetry.telemetry import StrataTelemetry
from fastapi import FastAPI

app = FastAPI()

@app.get("/metrics")
async def metrics():
    telemetry = StrataTelemetry.get_or_create()
    system_stats, per_tier_stats = telemetry.get_stats()
    return {
        "system_stats": system_stats.__dict__,
        "tiers": {tier.name: stats.__dict__ for tier, stats in per_tier_stats.items()},
    }
