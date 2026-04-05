import asyncio
from models import DispatchGridAction
from client import DispatchGridEnv

async def main():
    # Connect to existing server
    env = DispatchGridEnv(base_url="http://localhost:8000")

    # Use as normal
    result = await env.reset()
    result = await env.step(DispatchGridAction(
        ambulance_units=1,
        police_units=4,
        fire_units=2,
        priority_level=3,
        backup_requested=False,
    ))

    print(f"Success! Reward: {result.reward}")

if __name__ == "__main__":
    asyncio.run(main())