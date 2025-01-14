from dotenv import load_dotenv
from fastapi import FastAPI

from src.extraction.variable_handler import ExtractionHandler
from src.generate.task import GenerationTask
from src.generate.retrieve import RetrievalTask
from src.api.data_extraction import DataExtractionRequest
from src.db.database import Database
from src.interfaces.factory import get_interface

load_dotenv()

app = FastAPI()


@app.post("/data_extraction")
async def data_extraction(request: DataExtractionRequest):
    """
    1) Generate a specialized extraction prompt for the user request.
    2) Create a batch of prompts (GenerationTask) for the LLM; no tokens used yet.
    3) Return an ensemble_id for subsequent retrieval.
    """
    # Build an ExtractionHandler solely to generate the final prompt format.
    extraction_handler = ExtractionHandler(request.variables)
    extraction_prompt = extraction_handler.generate_extraction_prompt(request.prompt)
    request.prompt = extraction_prompt

    # Create the generation task and run it
    generation_task = GenerationTask()
    await generation_task.run(request)

    # Optionally, store dataset_path in the DB if needed for later evaluation
    db = Database()
    db.cursor.execute(
        """
        UPDATE batch_info
        SET dataset_path = ?
        WHERE ensemble_id = ?
        """,
        (request.dataset_path, generation_task.ensemble_id)
    )
    db.conn.commit()

    return {"ensemble_id": generation_task.ensemble_id}


@app.get("/retrieve/{ensemble_id}")
async def retrieve(ensemble_id: str):
    """
    Handles asynchronous retrieval of batch results:
    1. If retrieval is not done, triggers retrieval and returns status
    2. If retrieval is done, returns consolidated results
    """
    db = Database()
    batch_info = db.retrieve_batch_info(ensemble_id)
    if not batch_info:
        return {"error": f"Ensemble ID '{ensemble_id}' not found in batch_info."}

    # Check if retrieval is complete
    if not db.is_retrieval_done(ensemble_id):
        retrieval_task = RetrievalTask()
        retrieval_task.variables = batch_info["variables"]
        await retrieval_task.run(ensemble_id)
        return {
            "status": "processing",
            "message": "Retrieval in progress. Please check again later."
        }

    # Retrieval is done - fetch results from DB
    doc_ids = batch_info["doc_ids"]
    results = {}
    for doc_id in doc_ids:
        row = db.retrieve_extraction_result(ensemble_id, doc_id)
        raw_response = row["extraction_text"] if row else None
        var_extractions = db.retrieve_variable_extractions(ensemble_id, doc_id)
        results[doc_id] = {
            "raw_response": raw_response,
            "parsed_extractions": var_extractions
        }

    return {
        "status": "complete",
        "ensemble_id": ensemble_id,
        "results": results
    }


@app.get("/evaluate/{ensemble_id}")
async def evaluate(ensemble_id: str):
    """
    Evaluate the accuracy of the extracted variables (compared to CSV).
    """
    db = Database()
    batch_info = db.retrieve_batch_info(ensemble_id)
    if not batch_info:
        return {"error": f"Ensemble ID '{ensemble_id}' not found"}

    # Ensure retrieval is done before evaluation
    if not db.is_retrieval_done(ensemble_id):
        return {"error": "Retrieval is not done yet, please retrieve first."}

    # Build an ExtractionHandler with the stored variables
    variables = batch_info["variables"]
    handler = ExtractionHandler(variables)

    # Retrieve the dataset_path
    dataset_path = batch_info.get("dataset_path")
    if not dataset_path:
        return {"error": "No dataset_path found in batch_info. Please ensure it is stored properly."}

    # Evaluate
    evaluation_result = handler.evaluate_extractions(
        ensemble_id=ensemble_id,
        dataset_path=dataset_path,
        id_column="ID",
        db=db
    )

    # "evaluation_result" now contains both "metrics" and "comparisons"
    metrics = evaluation_result["metrics"]
    comparisons = evaluation_result["comparisons"]

    return {
        "status": "complete",
        "accuracy_report": metrics,
        "comparisons": comparisons
    }


@app.get("/batch_status/{ensemble_id}")
async def batch_status(ensemble_id: str):
    """
    Provide the status of each batch associated with a particular ensemble ID.
    This endpoint queries the model provider to retrieve the actual batch status.
    """
    db = Database()
    batch_info = db.retrieve_batch_info(ensemble_id)
    if not batch_info:
        return {"error": f"Ensemble ID '{ensemble_id}' not found in batch_info."}

    model = batch_info["model"]
    interface = get_interface(model)
    batch_ids = batch_info["batch_ids"]

    # Retrieve the batch objects and collect their statuses
    batches = await interface.retrieve_batches(batch_ids)
    statuses = {b.id: b.status for b in batches}

    return {
        "ensemble_id": ensemble_id,
        "batch_statuses": statuses
    }