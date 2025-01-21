import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI, APIError, RateLimitError
from pydantic import BaseModel

from src.interfaces.interface import (
    MAX_COMPLETION_TOKENS,
    BatchInterface,
    BatchResponseInterface,
    Interface,
    OPENAI_models_01,
)
from src.db.database import MongoDB

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# O1 specific rate limits
O1_RPM = 1000  # requests per minute
O1_TPM = 30_000_000  # tokens per minute

# Batch status constants
BATCH_STATUS_PENDING = "pending"
BATCH_STATUS_PROCESSING = "processing"
BATCH_STATUS_COMPLETED = "completed"
BATCH_STATUS_FAILED = "failed"

class TokenBucket:
    """Token bucket for rate limiting"""
    def __init__(self, tokens_per_minute: int):
        self.capacity = tokens_per_minute
        self.tokens = tokens_per_minute
        self.last_updated = time.time()
        self.tokens_per_second = tokens_per_minute / 60

    async def consume(self, tokens: int) -> bool:
        now = time.time()
        # Refill tokens based on time passed
        time_passed = now - self.last_updated
        self.tokens = min(
            self.capacity,
            self.tokens + time_passed * self.tokens_per_second
        )
        self.last_updated = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        else:
            # Calculate wait time needed for enough tokens
            wait_time = (tokens - self.tokens) / self.tokens_per_second
            logger.warning(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
            self.tokens = 0
            self.last_updated = time.time()
            return True

class O1Interface(Interface):
    client: AsyncOpenAI
    available_models = OPENAI_models_01
    
    def __init__(self, model: str | None = None, use_mongo: bool = True):
        super().__init__(model)
        self.client = AsyncOpenAI()
        self.db = MongoDB() if use_mongo else None
        # Set rate limits
        self.rpm_limit = O1_RPM
        self.request_interval = 60 / self.rpm_limit  # seconds between requests
        self.last_request_time = 0
        self.token_bucket = TokenBucket(O1_TPM)
        logger.info(f"Initialized O1Interface with model {model}")

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text.split()) * 1.3  # rough estimate

    async def wait_for_rate_limit(self):
        """Handle both RPM and TPM rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            wait_time = self.request_interval - time_since_last
            logger.debug(f"Rate limit: waiting {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
        self.last_request_time = time.time()

    async def create_batch(
        self, prompts: List[str], prompts_ids: List[str], prompt_run_id: str, seed: int
    ) -> str:
        """Creates a batch of prompts to be processed"""
        # Validate prompts
        if not prompts:
            raise ValueError("Cannot create batch with empty prompt list")
        
        for i, prompt in enumerate(prompts):
            if not prompt or not prompt.strip():
                raise ValueError(f"Prompt at index {i} is empty or contains only whitespace")

        # Estimate total tokens needed
        total_tokens = sum(self.estimate_tokens(p) for p in prompts) + len(prompts) * MAX_COMPLETION_TOKENS
        await self.token_bucket.consume(total_tokens)
        
        batch_id = f"batch_{int(time.time())}_{prompt_run_id}"
        
        # Create a directory for batch files if it doesn't exist
        batch_dir = Path("batch_files")
        batch_dir.mkdir(exist_ok=True)
        
        # Create input file for batch processing
        input_file = batch_dir / f"o1_batch_{batch_id}.json"
        batch_data = {
            "prompts": [
                {
                    "custom_id": prompts_ids[i],
                    "messages": [{"role": "user", "content": prompt}],
                    "model": self.model,
                    "max_completion_tokens": MAX_COMPLETION_TOKENS,
                    "seed": seed,
                }
                for i, prompt in enumerate(prompts)
            ]
        }
        with open(input_file, 'w') as f:
            json.dump(batch_data, f)
            
        # Create batch interface object
        batch = BatchInterface(
            id=batch_id,
            status=BATCH_STATUS_PENDING,
            created_at=int(time.time()),
            completed_at=None,
            output_file_id="",
            metadata={
                "prompt_run_id": str(prompt_run_id),
                "total_prompts": str(len(prompts)),
                "input_file": str(input_file),
                "model": self.model
            }
        )
        
        # Save batch status to MongoDB
        if self.db:
            self.db.save_batch({
                "ensemble_id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed_at": batch.completed_at,
                "output_file_id": batch.output_file_id,
                "metadata": batch.metadata,
                "prompt_run_id": prompt_run_id  # Add this to link with RetrievalTask
            })
        
        logger.info(f"O1 Batch created: {batch.id} with {len(prompts)} prompts")
        
        # Automatically process the batch
        try:
            await self.process_batch(batch_id)
        except Exception as e:
            logger.error(f"Failed to process batch {batch_id}: {str(e)}")
            # Don't raise here - let the batch be created even if processing fails
            
        return batch_id

    async def process_batch(self, batch_id: str) -> None:
        """Process and monitor a batch"""
        if not self.db:
            raise ValueError("MongoDB connection required for batch processing")
            
        # Get batch data
        batch_data = self.db.get_batch(batch_id)
        if not batch_data:
            raise ValueError(f"Batch {batch_id} not found")
            
        logger.info(f"Processing batch {batch_id} with data: {batch_data}")
            
        # Get input file path
        input_file = Path(batch_data["metadata"]["input_file"])
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Read prompts from input file
        with open(input_file, 'r') as f:
            data = json.load(f)
            prompts = data["prompts"]
            
        logger.info(f"Found {len(prompts)} prompts to process")
            
        # Update status to processing
        self.db.update_batch_status(
            batch_id,
            status=BATCH_STATUS_PROCESSING
        )
        
        try:
            # Process prompts in parallel
            async def process_prompt(prompt_data):
                try:
                    await self.wait_for_rate_limit()
                    response = await self.client.chat.completions.create(
                        model=prompt_data["model"],
                        messages=prompt_data["messages"],
                        max_completion_tokens=prompt_data["max_completion_tokens"],
                        seed=prompt_data["seed"]
                    )
                    logger.info(f"Successfully processed prompt {prompt_data['custom_id']}")
                    return {
                        "custom_id": prompt_data["custom_id"],
                        "response": {
                            "body": response.choices[0].message.content
                        },
                        "error": None
                    }
                except Exception as e:
                    logger.error(f"Error processing prompt {prompt_data['custom_id']}: {str(e)}")
                    return {
                        "custom_id": prompt_data["custom_id"],
                        "response": None,
                        "error": str(e)
                    }
            
            # Process all prompts in parallel with a semaphore to limit concurrency
            sem = asyncio.Semaphore(30)  # Process max 10 prompts at a time
            async def process_with_semaphore(prompt_data):
                async with sem:
                    return await process_prompt(prompt_data)
            
            results = await asyncio.gather(*[process_with_semaphore(prompt_data) for prompt_data in prompts])
            logger.info(f"Processed {len(results)} prompts")
            
            # Save results
            output_file = Path("batch_files") / f"o1_results_{batch_id}.json"
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f)
            logger.info(f"Saved results to {output_file}")
                
            # Update batch status with absolute path for reliable retrieval
            self.db.update_batch_status(
                batch_id,
                status=BATCH_STATUS_COMPLETED,
                completed_at=int(time.time()),
                output_file_id=str(output_file.absolute())
            )
            logger.info(f"Updated batch {batch_id} status to completed")
            
            # Clean up input file
            try:
                input_file.unlink()
                logger.info(f"Cleaned up input file {input_file}")
            except Exception as e:
                logger.warning(f"Could not delete input file: {e}")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            self.db.update_batch_status(
                batch_id,
                status=BATCH_STATUS_FAILED,
                error={"message": str(e)}
            )
            raise

    async def retrieve_batches(self, batch_ids: List[str]) -> List[BatchInterface]:
        """Retrieve the status of multiple batches"""
        if not self.db:
            raise ValueError("MongoDB connection required for batch retrieval")
            
        batches = []
        for batch_id in batch_ids:
            batch_data = self.db.get_batch(batch_id)
            if batch_data:
                logger.info(f"Retrieved batch data: {batch_data}")
                batch = BatchInterface(
                    id=batch_data["ensemble_id"],
                    status=batch_data["status"],
                    created_at=batch_data["created_at"],
                    completed_at=batch_data.get("completed_at"),
                    output_file_id=batch_data.get("output_file_id", ""),
                    metadata=batch_data.get("metadata", {})
                )
                batches.append(batch)
            else:
                logger.warning(f"Batch {batch_id} not found")
                
        return batches

    async def download_batches(
        self, batches: List[BatchInterface]
    ) -> List[List[BatchResponseInterface]]:
        """Download results of completed batches"""
        all_results = []
        for batch in batches:
            if not batch.output_file_id:
                logger.warning(f"No output file ID for batch {batch.id}")
                all_results.append([])
                continue
                
            try:
                # Use absolute path from MongoDB
                output_file = Path(batch.output_file_id)
                if not output_file.exists():
                    logger.warning(f"Output file not found: {output_file}")
                    all_results.append([])
                    continue
                    
                logger.info(f"Reading results from {output_file}")
                with open(output_file, 'r') as f:
                    results = json.load(f)
                    
                batch_results = []
                for result in results:
                    if result.get("error"):
                        logger.warning(f"Error in batch response: {result['error']}")
                        continue
                        
                    if not result.get("response") or not result["response"].get("body"):
                        logger.warning(f"Missing response body for {result.get('custom_id')}")
                        continue
                        
                    batch_results.append(
                        BatchResponseInterface(
                            id=result["custom_id"],
                            response=result["response"]["body"]
                        )
                    )
                logger.info(f"Processed {len(batch_results)} results from batch {batch.id}")
                all_results.append(batch_results)
                
                # Clean up output file
                try:
                    output_file.unlink()
                    logger.info(f"Cleaned up output file {output_file}")
                except Exception as e:
                    logger.warning(f"Could not delete output file: {e}")
                    
            except Exception as e:
                logger.error(f"Error downloading batch {batch.id}: {str(e)}")
                all_results.append([])
                
        return all_results 