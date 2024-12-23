from dotenv import load_dotenv
from fastapi import FastAPI

from src.generate.task import GenerationTask
from src.generate.retrieve import RetrievalTask

from src.api.data_extraction import DataExtractionRequest


load_dotenv()

app = FastAPI()


@app.post("/data_extraction")
async def data_extraction(request: DataExtractionRequest):
    generation_task = GenerationTask()
    await generation_task.run(request)
    return {"ensemble_id": generation_task.ensemble_id}


@app.get("/retrieve/{ensemble_id}")
async def retrieve(ensemble_id: str):
    retrieval_task = RetrievalTask()
    results = await retrieval_task.run(ensemble_id)
    return {"results": results}
