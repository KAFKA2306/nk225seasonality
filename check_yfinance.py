import asyncio
from datetime import datetime, timedelta

from src.config import config
from src.data.ingestion import DataIngestionPipeline


async def main():
    pipeline = DataIngestionPipeline(config)
    end = datetime.now()
    start = end - timedelta(days=30)
    df = await pipeline.collect_data(start, end)
    print(f"Fetched {len(df)} rows")
    print(df.head())


if __name__ == "__main__":
    asyncio.run(main())
