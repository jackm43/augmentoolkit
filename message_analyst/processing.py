import random
import traceback
from augmentoolkit.generation_functions.engine_wrapper_class import EngineWrapper
from augmentoolkit.utils.write_output_to_file import write_output_to_file
from message_analyst.steps import API_KEY_A, API_KEY_B, BASE_URL_A, BASE_URL_B, CONCURRENCY_LIMIT, LOGICAL_MODEL_A, LOGICAL_MODEL_B, MODE_A, MODE_B, add_key, chunking_algorithm, count_tokens, make_id

import nltk
from tqdm import asyncio as tqdmasyncio
import asyncio
import glob
import logging
import os
import sys
import time
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('message_analyst.log')
    ]
)

config_path = os.environ["CONFIG_PATH"]
try:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
except Exception as e:
    logging.error(f"Failed to load config from {config_path}: {str(e)}")
    sys.exit(1)

# Configuration
WORK_IN_PHASES = bool(config["PHASES"]["WORK_IN_PHASES"])
PHASE_INDEX = int(config["PHASES"]["PHASE_INDEX"])
USE_SUBSET = bool(config["SYSTEM"]["USE_SUBSET"])
SUBSET_SIZE = int(config["SYSTEM"]["SUBSET_SIZE"])
CHUNK_SIZE = int(config["SYSTEM"]["CHUNK_SIZE"])
INPUT = os.path.abspath(config["PATH"]["INPUT"])

async def main():
    try:
        logging.info("Starting message analysis pipeline")
        start_time = time.time()

        # Set up rate-limit-conscious functions
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        async def run_task_with_limit(task):
            async with semaphore:
                return await task

        # Find input files
        extensions = [".txt", ".md"]
        source_texts = []
        for extension in extensions:
            path = os.path.join(INPUT, f"**/*{extension}")
            source_texts.extend(glob.glob(path, recursive=True))

        if not source_texts:
            logging.error(f"No source texts found in: {INPUT}")
            return
        
        logging.info(f"Found {len(source_texts)} source files: {source_texts}")

        # Initialize engine wrappers
        try:
            engine_wrapper = EngineWrapper(
                model=LOGICAL_MODEL_A,
                api_key=API_KEY_A,
                base_url=BASE_URL_A,
                mode=MODE_A,
            )
            
            engine_wrapper_large = EngineWrapper(
                model=LOGICAL_MODEL_B,
                api_key=API_KEY_B,
                base_url=BASE_URL_B,
                mode=MODE_B,
            )
        except Exception as e:
            logging.error(f"Failed to initialize engine wrappers: {str(e)}")
            return

        # Process chunks
        sentence_chunks = []
        for source_text in source_texts:
            try:
                chunks = chunking_algorithm(source_text, max_token_length=CHUNK_SIZE)
                sentence_chunks.extend(chunks)
                logging.info(f"Processed {source_text}: {len(chunks)} chunks")
            except Exception as e:
                logging.error(f"Failed to process {source_text}: {str(e)}")
                continue

        if USE_SUBSET:
            logging.info(f"Using subset of {SUBSET_SIZE} chunks for testing")
            sentence_chunks = random.sample(sentence_chunks, min(SUBSET_SIZE, len(sentence_chunks)))

        # Generate data
        output_list = []
        data_generations_tasks = [
            add_key(
                input_data=chunk,
                engine_wrapper=engine_wrapper_large,
                idx=idx,
                output_list=output_list
            ) for idx, chunk in enumerate(sentence_chunks)
        ]

        try:
            coroutines = [run_task_with_limit(task) for task in data_generations_tasks]
            for future in tqdmasyncio.tqdm.as_completed(coroutines):
                try:
                    await future
                except Exception as e:
                    logging.error(f"Task failed: {str(e)}")
                    continue
        except Exception as e:
            logging.error(f"Failed during data generation: {str(e)}")
            return

        elapsed_time = time.time() - start_time
        logging.info(f"Processing completed in {elapsed_time:.2f} seconds")
        
        if output_list:
            logging.info(f"Generated {len(output_list)} outputs")
            logging.info(f"Sample output: {output_list[0]}")
        else:
            logging.warning("No outputs were generated")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
