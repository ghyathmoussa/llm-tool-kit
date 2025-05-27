import json
import logging
import os
from pathlib import Path
from typing import Generator, Optional
import gc  # For garbage collection
from tqdm import tqdm
from config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)

def get_data(path: str):
    """
    DEPRECATED: Use get_data_iter for large files instead
    """
    logging.warning("get_data() loads the entire file into memory. For large files, use get_data_iter() instead.")
    if not path:
        path = f"/app/data/{path}"
    
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def get_data_iter(path: str, batch_size: Optional[int] = None) -> Generator:
    """
    Loads data iteratively from a JSON file (line-delimited JSON).
    Memory-efficient version that optionally yields data in batches.
    
    Args:
        path: Relative path to the data file within the data directory.
        batch_size: If provided, yields data in batches of this size.
    """
    data_path = f"/app/data/{path}"
    
    logging.info(f"Loading data from: {data_path}")
    
    start = 0
    end = batch_size
    stop = False
    try:
        if batch_size:
            # Batch processing mode
            batch = []
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        if 'text' in line and not stop:
                            record = json.loads(line)[0]
                            for i in range(start, end):
                                batch.append(record['text'])
                                yield batch
                                batch = []
                                start = start + batch_size
                                end = end + batch_size
                                if end > len(record):
                                    stop = True
                                gc.collect()  # Force garbage collection
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping invalid JSON line: {line[:50]}... - Error: {e}")
                
                # Yield remaining batch if not empty
                if batch:
                    yield batch
        else:
            # Single record mode
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if 'text' in line:
                            records = json.loads(line)
                            print(len(records))
                            print(type(records))
                            for rec in records:
                                yield rec['text']
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping invalid JSON line: {line[:50]}... - Error: {e}")
    
    except FileNotFoundError:
        logging.error(f"Data file not found at: {data_path}")
        return

if __name__ == "__main__":
    file_rel_path = "data1.jsonl"
    data_path = PROJECT_ROOT / "data" / file_rel_path
    logging.info(f"Processing data from: {data_path}")
    
    count = 0
    try:
        # Get file size for progress bar
        file_size = os.path.getsize(data_path)
        
        # Process in batches of 1000 to save memory
        batch_size = 1000
        batch_count = 0
        
        for batch in get_data_iter(file_rel_path, batch_size=batch_size):
            batch_count += 1
            count += len(batch)
            logging.info(f"Processed batch {batch_count} with {len(batch)} records. Total: {count}")
            
            # Optional: stop after processing some batches (for testing)
            if batch_count >= 10:  # Process only 10 batches for testing
                break
                
        logging.info(f"Finished processing. Total valid records processed: {count}")
        
    except FileNotFoundError:
        logging.error(f"Data file not found at: {data_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
