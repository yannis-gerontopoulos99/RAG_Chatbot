import asyncio
import json
from rag_pipeline import RAGPipeline  # assuming your class file is saved as rag_pipeline.py

async def main():
    # Create pipeline instance
    rag = RAGPipeline()

    # Initialize models and load/create index
    rag.initialize()

    # Ask a question
    result = rag.process_query("Can satelites help observe wildfires?")
    
    # Pretty-print answer
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
