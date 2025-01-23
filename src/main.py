from dotenv import load_dotenv
from fastapi import FastAPI

from src.extraction.variable_handler import ExtractionHandler, VariableDefinition
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
    2) Create one or more prompt_run_ids (GenerationTask) for the LLM.
    3) Return a dictionary with ensemble_id, sub_ensemble_id, chunk runs, etc.

    ID levels: ensemble_id > sub_ensemble_id > prompt_run_id
    """
    generation_task = GenerationTask()
    result_dict = await generation_task.run(request)

    # Optionally, store dataset_path for each created prompt_run_id if chunked,
    db = Database()

    # If chunk logic returns multiple, handle them:
    created_runs = result_dict.get("created_prompt_runs", [])
    if created_runs:
        for pr_info in created_runs:
            pr_id = pr_info["prompt_run_id"]
            db.cursor.execute(
                """
                UPDATE batch_info
                SET dataset_path = ?
                WHERE prompt_run_id = ?
                """,
                (request.dataset_path, pr_id)
            )
        db.conn.commit()

    return result_dict

@app.get("/retrieve/{ensemble_id}")
async def retrieve(ensemble_id: str):
    """
    Handles asynchronous retrieval of batch results for all prompt_run_ids in the ensemble.
    1. For each prompt_run_id in the ensemble, retrieves data from the model if not already done
    2. After all retrievals complete, merges data across all prompt_run_ids
    """
    retrieval_task = RetrievalTask()
    prompt_runs = retrieval_task.db.find_prompt_runs_for_ensemble(ensemble_id)
    
    # First ensure all prompt_runs have been retrieved from the model
    for pr in prompt_runs:
        # This ensures data is actually fetched for each run
        await retrieval_task.run(pr)
    
    # Now assemble across everything
    final_data = await retrieval_task._assemble_results(ensemble_id)
    return final_data

@app.get("/evaluate/{ensemble_id}")
async def evaluate(ensemble_id: str, use_ai_evaluation: bool = False):
    """
    Evaluate the accuracy of the extracted variables (compared to CSV).
    Optional parameter use_ai_evaluation switches between regular and AI-based evaluation.
    """
    retrieval_task = RetrievalTask()
    final_data = await retrieval_task._assemble_results(ensemble_id)

    if final_data["status"] == "no_data":
        return {
            "status": "no_data",
            "accuracy_report": {},
            "comparisons": []
        }

    # Load the master variable definitions
    db = retrieval_task.db
    master_vars = db.retrieve_master_variables(ensemble_id) or []
    all_defs = [VariableDefinition(**mv) for mv in master_vars]

    # Get dataset path and context information
    prompt_runs = db.find_prompt_runs_for_ensemble(ensemble_id)
    if not prompt_runs:
        return {
            "status": "no_data",
            "accuracy_report": {},
            "comparisons": []
        }

    # Get batch info from the first prompt run
    batch_info = db.retrieve_batch_info(prompt_runs[0])
    if not batch_info:
        return {
            "status": "no_batch_info",
            "accuracy_report": {},
            "comparisons": []
        }

    dataset_path = batch_info.get("dataset_path")
    if not dataset_path:
        return {
            "status": "no_dataset_path",
            "accuracy_report": {},
            "comparisons": []
        }

    # Get the ID column from batch info
    id_column = batch_info.get("id_column", "id")  # Default to "id" if not specified

    # Prepare context with safe dictionary access
    context = {
        "preprompt": batch_info.get("preprompt", ""),
        "variable_info": batch_info.get("variable_info", "")
    }

    if use_ai_evaluation:
        # Use AI-based evaluation
        from src.evaluation.ai_evaluator import AIEvaluator
        evaluator = AIEvaluator(all_defs, ensemble_id)
        evaluation_result = await evaluator.evaluate_extractions(
            merged_data=final_data["results"],
            dataset_path=dataset_path,
            id_column=id_column,
            context=context
        )
    else:
        # Use regular evaluation
        handler = ExtractionHandler(
            variable_definitions=all_defs,
            ensemble_id=ensemble_id,
            db=db
        )
        evaluation_result = handler.evaluate_ensemble_extractions(
            merged_data=final_data["results"],
            dataset_path=dataset_path,
            id_column=id_column,
            db=db
        )

    return {
        "status": "complete",
        "accuracy_report": evaluation_result["metrics"],
        "comparisons": evaluation_result["comparisons"],
        "evaluation_type": "ai" if use_ai_evaluation else "regular"
    }

#@app.get("/batch_status/{ensemble_id}")
#async def batch_status(ensemble_id: str):
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

    # Retrieve batch objects and collect statuses
    batches = await interface.retrieve_batches(batch_ids)
    statuses = {b.id: b.status for b in batches}

    return {
        "ensemble_id": ensemble_id,
        "batch_statuses": statuses
    }
