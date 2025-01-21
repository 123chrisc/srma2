import asyncio
import json
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from src.main import app
from src.interfaces.openai_01 import (
    O1Interface,
    BATCH_STATUS_COMPLETED,
    BATCH_STATUS_FAILED
)

# Load environment variables
load_dotenv(override=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create test client
client = TestClient(app)

async def test_batch_processing():
    """Test the batch processing implementation"""
    try:
        # Initialize interface
        interface = O1Interface("o1-2024-12-17", use_mongo=True)
        
        # Test data
        prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?"
        ]
        prompt_ids = [f"test_prompt_{i}" for i in range(len(prompts))]
        
        logger.info("Step 1: Creating batch...")
        batch_id = await interface.create_batch(
            prompts=prompts,
            prompts_ids=prompt_ids,
            prompt_run_id="test_batch_1",
            seed=42
        )
        logger.info(f"Created batch with ID: {batch_id}")
        
        # Verify batch files were created
        batch_dir = Path("batch_files")
        batch_dir.mkdir(exist_ok=True)
        input_file = batch_dir / f"o1_batch_{batch_id}.json"
        assert input_file.exists(), f"Input file not created: {input_file}"
        
        # Verify file contents
        with open(input_file, 'r') as f:
            data = json.load(f)
            assert "prompts" in data, "Missing prompts array in batch file"
            assert len(data["prompts"]) == len(prompts), f"Expected {len(prompts)} prompts, got {len(data['prompts'])}"
            for prompt_data in data["prompts"]:
                assert "custom_id" in prompt_data, "Missing custom_id in prompt data"
                assert "messages" in prompt_data, "Missing messages in prompt data"
                assert "model" in prompt_data, "Missing model in prompt data"
                assert "max_completion_tokens" in prompt_data, "Missing max_completion_tokens in prompt data"
                assert "seed" in prompt_data, "Missing seed in prompt data"
        
        logger.info("Step 2: Processing batch...")
        await interface.process_batch(batch_id)
        
        logger.info("Step 3: Retrieving batch status...")
        batches = await interface.retrieve_batches([batch_id])
        assert len(batches) == 1, f"Expected 1 batch, got {len(batches)}"
        batch = batches[0]
        
        logger.info(f"Batch status: {batch.status}")
        if batch.status == BATCH_STATUS_COMPLETED:
            logger.info("Step 4: Downloading results...")
            results = await interface.download_batches(batches)
            assert len(results) == 1, f"Expected 1 result set, got {len(results)}"
            batch_results = results[0]
            assert len(batch_results) == len(prompts), f"Expected {len(prompts)} results, got {len(batch_results)}"
            
            # Print sample results
            for i, result in enumerate(batch_results):
                logger.info(f"\nPrompt {i+1}: {prompts[i]}")
                logger.info(f"Response: {result.response}")
        elif batch.status == BATCH_STATUS_FAILED:
            logger.error(f"Batch processing failed")
        else:
            logger.info(f"Batch is in status: {batch.status}")
            
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

async def test_parallel_processing():
    """Test parallel processing of prompts with timing information"""
    try:
        interface = O1Interface("o1-2024-12-17", use_mongo=True)
        
        # Create a larger batch of prompts
        prompts = [
            f"What is {i} + {i}?" for i in range(10)  # 10 simple math questions
        ]
        prompt_ids = [f"test_prompt_{i}" for i in range(len(prompts))]
        
        logger.info(f"Testing parallel processing with {len(prompts)} prompts...")
        
        # Time the batch creation
        start_time = time.time()
        batch_id = await interface.create_batch(
            prompts=prompts,
            prompts_ids=prompt_ids,
            prompt_run_id="test_parallel",
            seed=42
        )
        create_time = time.time() - start_time
        logger.info(f"Batch creation took {create_time:.2f} seconds")
        
        # Time the batch processing
        start_time = time.time()
        await interface.process_batch(batch_id)
        process_time = time.time() - start_time
        logger.info(f"Batch processing took {process_time:.2f} seconds")
        
        # Calculate average time per prompt
        avg_time = process_time / len(prompts)
        logger.info(f"Average time per prompt: {avg_time:.2f} seconds")
        
        # Verify results
        batches = await interface.retrieve_batches([batch_id])
        batch = batches[0]
        
        if batch.status == BATCH_STATUS_COMPLETED:
            results = await interface.download_batches(batches)
            batch_results = results[0]
            
            # Count successful responses
            successful = sum(1 for r in batch_results if r.response)
            logger.info(f"Successfully processed {successful}/{len(prompts)} prompts")
            
            # Print timing summary
            logger.info("\nTiming Summary:")
            logger.info(f"Total prompts: {len(prompts)}")
            logger.info(f"Batch creation time: {create_time:.2f}s")
            logger.info(f"Batch processing time: {process_time:.2f}s")
            logger.info(f"Average time per prompt: {avg_time:.2f}s")
            logger.info(f"Throughput: {len(prompts)/process_time:.2f} prompts/second")
            
        else:
            logger.error(f"Batch processing failed: {batch.error}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

async def main():
    """Run all tests"""
    await test_batch_processing()  # Run the original test
    logger.info("\n" + "="*50 + "\n")
    await test_parallel_processing()  # Run the parallel processing test

if __name__ == "__main__":
    asyncio.run(main()) 