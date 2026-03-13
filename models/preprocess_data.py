import argparse
import json
import os
from typing import Dict, Generator, List, Optional

import trankit
from tqdm import tqdm
from transformers import AutoTokenizer

from models.db_manager import PostgreSQLManager
from utils.logger import setup_app_logger

# --- Configuration ---
INPUT_FOLDER_PATH = "../source_data/"
OUTPUT_JSONL_PATH = "../data/processed_data.jsonl"
MAX_TOKENS_PER_CHUNK = 2048 # As per your requirement
FIRST_LINE_TO_SKIP = "هذا الملف آليا بواسطة المكتبة الشاملة"

logger = setup_app_logger(__name__)

class Processor:
    def __init__(self, model_name: str, max_tokens: int, db_manager: Optional[PostgreSQLManager] = None):
        """
        Initialize the Processor with optional database manager.

        Args:
            model_name: Name of the tokenizer model to use
            max_tokens: Maximum tokens per chunk
            db_manager: Optional PostgreSQLManager for database operations
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.db_manager = db_manager
        # Initialize Trankit for Arabic
        self.nlp = trankit.Pipeline('arabic', cache_dir='../cache')

    def _split_text_to_fit(self, text, max_len):
        """
        Splits a given text into smaller pieces that are_ideally_ less than
        or equal to max_len tokens. Splits are attempted at sentence boundaries first,
        then word boundaries.
        A single word exceeding max_len will be returned as its own piece (and will be oversized).
        """
        # Try to split into sentences first using Trankit
        sentences = self.sentencize_text(text)
        
        output_pieces = []
        for sent in sentences:
            if not sent.strip(): # Skip empty sentences
                continue
            
            _, sent_tokens = self.tokenize_and_count(sent)
            if sent_tokens <= max_len:
                output_pieces.append(sent)
            else:
                # Sentence is too long, split by words
                logger.info(f"Sentence part is too long ({sent_tokens} tokens), splitting by words: '{sent[:100]}...'")
                words = sent.split() 
                if not words:
                    continue

                current_piece_text_for_word_split = ""
                for word_idx, word in enumerate(words):
                    text_to_try_adding = (" " + word) if current_piece_text_for_word_split else word
                    
                    prospective_new_piece = current_piece_text_for_word_split + text_to_try_adding
                    _, prospective_tokens = self.tokenize_and_count(prospective_new_piece)

                    if prospective_tokens <= max_len:
                        current_piece_text_for_word_split = prospective_new_piece
                    else:
                        if current_piece_text_for_word_split.strip():
                            output_pieces.append(current_piece_text_for_word_split)
                        
                        _, single_word_tokens = self.tokenize_and_count(word)
                        if single_word_tokens > max_len:
                            logger.warning(f"Word '{word[:50]}...' ({single_word_tokens} tokens) itself exceeds max_len ({max_len}). Adding as oversized piece.")
                            if word.strip(): output_pieces.append(word)
                            current_piece_text_for_word_split = "" 
                        else:
                            current_piece_text_for_word_split = word
                
                if current_piece_text_for_word_split.strip():
                    output_pieces.append(current_piece_text_for_word_split)
                    
        return [p for p in output_pieces if p.strip()]

    def tokenize_and_count(self, text_batch):
        """
        Tokenizer Function to return list of token counts for a batch of text
        """
        tokens = self.tokenizer.encode(text_batch)
        return tokens, len(tokens)

    # --- Helper Functions ---
    def _process_single_text(self, text: str, fatwa_id: int, chunk_index_offset: int = 0) -> List[Dict]:
        """
        Process a single text and return chunks for database insertion.

        Args:
            text: The text to chunk
            fatwa_id: The fatwa ID for the text
            chunk_index_offset: Starting chunk index (usually 0)

        Returns:
            List of chunk dictionaries ready for database insertion
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        all_chunks = []
        current_chunk_parts = []
        current_chunk_tokens = 0
        chunk_index = chunk_index_offset

        # Split into paragraphs and sentences
        paragraphs = [p for p in text.split("\n") if p.strip()]

        for paragraph in paragraphs:
            sentences = self.sentencize_text(paragraph)

            for sentence in sentences:
                if not sentence.strip():
                    continue

                # Check if sentence needs splitting
                _, sentence_tokens = self.tokenize_and_count(sentence)

                if sentence_tokens > self.max_tokens:
                    logger.warning(f"Sentence too long ({sentence_tokens} tokens), splitting for fatwa_id={fatwa_id}")
                    sentence_parts = self._split_text_to_fit(sentence, self.max_tokens)
                else:
                    sentence_parts = [sentence]

                for sentence_part in sentence_parts:
                    if not sentence_part.strip():
                        continue

                    _, part_tokens = self.tokenize_and_count(sentence_part)

                    if part_tokens == 0:
                        continue

                    # Check if adding to current chunk would exceed limit
                    if current_chunk_tokens + part_tokens > self.max_tokens and current_chunk_parts:
                        # Finalize current chunk
                        chunk_text = "\n".join(current_chunk_parts)
                        all_chunks.append({
                            'fatwa_id': fatwa_id,
                            'chunk_index': chunk_index,
                            'chunk_text': chunk_text,
                            'word_count': current_chunk_tokens,  # Using token count as word_count
                            'embedding_status': 'pending'
                        })
                        chunk_index += 1

                        # Start new chunk
                        current_chunk_parts = [sentence_part]
                        current_chunk_tokens = part_tokens
                    else:
                        # Add to current chunk
                        current_chunk_parts.append(sentence_part)
                        current_chunk_tokens += part_tokens

        # Add final chunk if any remaining
        if current_chunk_parts:
            chunk_text = "\n".join(current_chunk_parts)
            all_chunks.append({
                'fatwa_id': fatwa_id,
                'chunk_index': chunk_index,
                'chunk_text': chunk_text,
                'word_count': current_chunk_tokens,
                'embedding_status': 'pending'
            })

        return all_chunks

    def clean_text(self, text):
        """Basic text cleaning."""
        text = text.strip()

        # remove the header line
        parts = text.split("\n\n", 1)
        if len(parts) > 1:
            text = parts[1]
        
        return text

    def sentencize_text(self, text):
        """
        Use Trankit to split text into sentences.
        """
        try:
            doc = self.nlp.ssplit(text)
            sentences = [sent['text'] for sent in doc['sentences']]
            return sentences
        except Exception as e:
            logger.warning(f"Error during sentence segmentation: {e}")
            # Fall back to simple splitting if Trankit fails
            return [text]

    def stream_semantic_units(self, file_path, line_to_skip=None):
        """
        Reads the input file and yields semantic units (sentences).
        Uses Trankit for sentence segmentation.
        Skips a specified first line if provided.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                logger.info(f"Reading file: {file_path}")
                text = f.read()
                text = self.clean_text(text)
                
                # First split by paragraphs to maintain some structure
                paragraphs = [p for p in text.split("\n") if p.strip()]
                print(len(paragraphs))
                for paragraph in tqdm(paragraphs, desc="Processing paragraphs"):
                    # Use Trankit to split paragraph into sentences
                    sentences = self.sentencize_text(paragraph)
                    for sentence in sentences:
                        if sentence.strip():
                            yield sentence.strip()
        except FileNotFoundError:
            logger.error(f"Error: Input file not found at {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading or processing file {file_path}: {e}")
            raise

    # --- Main Processing Logic ---
    def process_folder(self, input_folder_path, output_jsonl_path, skip_header_line=None):
        """
        Processes all .txt files in a given folder, chunks them, and writes to a single JSONL file.
        """
        all_chunks_data = []
        chunk_id_counter = 0

        logger.info(f"Searching for .txt files in '{input_folder_path}'...")
        try:
            txt_files = [f for f in os.listdir(input_folder_path) if f.endswith('.txt') and os.path.isfile(os.path.join(input_folder_path, f))]
        except FileNotFoundError:
            logger.error(f"Error: Input directory not found at {input_folder_path}")
            return
        
        logger.info(f"Found {len(txt_files)} .txt files to process.")

        for filename in tqdm(txt_files, desc="Processing files"):
            file_path = os.path.join(input_folder_path, filename)
            file_chunks, updated_chunk_id_counter = self._process_single_file(
                input_file_path=file_path,
                chunk_id_offset=chunk_id_counter,
                skip_header_line=skip_header_line
            )
            all_chunks_data.extend(file_chunks)
            chunk_id_counter = updated_chunk_id_counter

        # Write to JSONL file
        try:
            with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
                for entry in tqdm(all_chunks_data, desc="Writing chunks to JSONL"):
                    json.dump(entry, outfile, ensure_ascii=False)
                    outfile.write('\n')
            logger.info(f"Successfully processed and wrote {len(all_chunks_data)} chunks from {len(txt_files)} file(s) to {output_jsonl_path}")
        except IOError:
            logger.error(f"Error: Could not write to output file {output_jsonl_path}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during writing: {e}")

    def process_database(
        self,
        text_column: str = 'answer',
        skip_existing: bool = True,
        delete_existing: bool = False,
        batch_size: int = 100,
        limit: Optional[int] = None
    ):
        """
        Process fatwas from PostgreSQL database and save chunks to database.

        Args:
            text_column: Name of column in fatwas table containing text to chunk
            skip_existing: Skip fatwas that already have chunks (default: True)
            delete_existing: Delete existing chunks before re-processing (conflicts with skip_existing)
            batch_size: Number of chunks to insert in one batch
            limit: Optional limit on number of fatwas to process
        """
        if not self.db_manager:
            raise ValueError("Database manager not initialized")

        if skip_existing and delete_existing:
            raise ValueError("Cannot use both skip_existing and delete_existing")

        logger.info(f"Starting database processing from column '{text_column}'")
        logger.info(f"Skip existing: {skip_existing}, Delete existing: {delete_existing}")

        # Initialize connection pool if not already done
        if not self.db_manager.connection_pool:
            self.db_manager.initialize_pool()

        # Get existing fatwa_ids if skipping
        existing_fatwa_ids = None
        if skip_existing:
            existing_fatwa_ids = self.db_manager.get_existing_fatwa_ids()
            logger.info(f"Found {len(existing_fatwa_ids)} fatwas with existing chunks")

        # Process fatwas
        chunks_batch = []
        total_fatwas_processed = 0
        total_chunks_attempted = 0
        total_chunks_inserted = 0
        skipped_count = 0

        fatwas_generator = self.db_manager.get_fatwas_to_process(
            text_column=text_column,
            limit=limit,
            skip_existing=skip_existing,
            existing_fatwa_ids=existing_fatwa_ids
        )

        for fatwa in tqdm(fatwas_generator, desc="Processing fatwas"):
            fatwa_id = fatwa['id']
            text = fatwa['text']

            # Skip if delete_existing is set and we need to clear first
            if delete_existing:
                self.db_manager.delete_chunks_for_fatwa(fatwa_id)

            # Process text into chunks
            chunks = self._process_single_text(
                text=text,
                fatwa_id=fatwa_id,
                chunk_index_offset=0
            )

            if not chunks:
                logger.warning(f"No chunks generated for fatwa_id={fatwa_id}")
                skipped_count += 1
                continue

            # Add chunks to batch
            chunks_batch.extend(chunks)
            total_fatwas_processed += 1

            # Insert batch if it reaches batch_size
            if len(chunks_batch) >= batch_size:
                attempted, inserted = self.db_manager.insert_chunks_batch(chunks_batch)
                total_chunks_attempted += attempted
                total_chunks_inserted += inserted
                duplicates = attempted - inserted
                logger.info(f"Inserted batch: {inserted} chunks (duplicates skipped: {duplicates}, total inserted: {total_chunks_inserted})")
                chunks_batch = []

        # Insert remaining chunks
        if chunks_batch:
            attempted, inserted = self.db_manager.insert_chunks_batch(chunks_batch)
            total_chunks_attempted += attempted
            total_chunks_inserted += inserted
            duplicates = attempted - inserted
            logger.info(f"Inserted final batch: {inserted} chunks (duplicates skipped: {duplicates}, total inserted: {total_chunks_inserted})")

        logger.info(f"Processing complete: {total_fatwas_processed} fatwas processed, {total_chunks_inserted} chunks inserted, {total_chunks_attempted - total_chunks_inserted} duplicates skipped, {skipped_count} fatwas with no chunks")

        # Close connection pool
        self.db_manager.close_pool()

    def _process_single_file(self, input_file_path, chunk_id_offset=0, skip_header_line=None):
        """
        Processes a single input file, chunks it, and returns chunk data.
        """
        all_chunks_data = []
        current_chunk_text_parts = []
        current_chunk_token_count = 0
        chunk_id_counter = chunk_id_offset

        logger.info(f"Starting processing for {input_file_path}...")

        semantic_unit_generator = self.stream_semantic_units(input_file_path, skip_header_line)

        for i, sentence in enumerate(tqdm(semantic_unit_generator, desc=f"Processing sentences for {os.path.basename(input_file_path)}")):
            if not sentence:
                continue

            # New adaptive splitting logic
            processed_parts_for_sentence = []
            _, initial_token_count = self.tokenize_and_count(sentence)

            if initial_token_count > self.max_tokens:
                logger.warning(f"Semantic unit {i+1} (starts with '{sentence[:50]}...') has {initial_token_count} tokens, exceeding max_tokens ({self.max_tokens}). Attempting to split adaptively.")
                processed_parts_for_sentence = self._split_text_to_fit(sentence, self.max_tokens)
                if not processed_parts_for_sentence:
                    logger.warning(f"Splitting semantic unit '{sentence[:50]}...' resulted in no processable parts.")
                    continue
            else:
                processed_parts_for_sentence = [sentence]
            
            for sentence_part in processed_parts_for_sentence:
                if not sentence_part.strip():
                    continue

                _, sentence_token_count = self.tokenize_and_count(sentence_part)

                if sentence_token_count == 0 and sentence_part:
                    logger.warning(f"Warning: Tokenizer returned 0 tokens for non-empty sentence part: '{sentence_part[:100]}...'")
                    continue
                elif sentence_token_count == 0:
                    continue

                # Handle oversized single sentence parts
                # This might happen if _split_text_to_fit returns a single word that's too long.
                if sentence_token_count > self.max_tokens:
                    logger.warning(f"Warning: Sentence part (from unit {i+1}, original: '{sentence[:50]}...') has {sentence_token_count} tokens, exceeding max_tokens ({self.max_tokens}). Part: '{sentence_part[:50]}...'")
                    # The chunking logic below will handle this part, potentially as an oversized chunk.

                # If adding the current sentence_part would exceed the max token limit for the current chunk
                if current_chunk_token_count + sentence_token_count > self.max_tokens and current_chunk_text_parts:
                    # Finalize the current chunk
                    final_chunk_text = "\n".join(current_chunk_text_parts)
                    chunk_id_counter += 1
                    json_object = {
                        "text": final_chunk_text,
                        "source_document": input_file_path,
                        "chunk_id": f"chunk_{chunk_id_counter}",
                        "token_count_estimate": current_chunk_token_count
                    }
                    all_chunks_data.append(json_object)

                    # Start a new chunk with the current sentence_part
                    current_chunk_text_parts = [sentence_part]
                    current_chunk_token_count = sentence_token_count
                else:
                    # Add the current sentence_part to the current chunk
                    current_chunk_text_parts.append(sentence_part)
                    current_chunk_token_count += sentence_token_count

        # Add the last remaining chunk
        if current_chunk_text_parts:
            final_chunk_text = "\n".join(current_chunk_text_parts)
            chunk_id_counter += 1
            json_object = {
                "text": final_chunk_text,
                "source_document": input_file_path,
                "chunk_id": f"chunk_{chunk_id_counter}",
                "token_count_estimate": current_chunk_token_count
            }
            all_chunks_data.append(json_object)
        
        return all_chunks_data, chunk_id_counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process text data from files or PostgreSQL database into chunks."
    )

    # Input mode selection
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-dir',
        type=str,
        default=INPUT_FOLDER_PATH,
        help='Path to folder containing input .txt files (file mode)'
    )
    input_group.add_argument(
        '--database-url',
        type=str,
        help='PostgreSQL connection URL (database mode), e.g., postgresql://user:pass@host:port/db'
    )

    # Output mode selection
    parser.add_argument(
        '--output-file',
        type=str,
        default=OUTPUT_JSONL_PATH,
        help='Path to output JSONL file (file mode only)'
    )

    # Common options
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=MAX_TOKENS_PER_CHUNK,
        help='Maximum tokens per chunk.'
    )
    parser.add_argument(
        '--skip-header',
        type=str,
        default=FIRST_LINE_TO_SKIP,
        help='First line to skip in input files (file mode only)'
    )

    # Database-specific options
    parser.add_argument(
        '--text-column',
        type=str,
        default='answer',
        help='Name of column in fatwas table to chunk (database mode, default: answer)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip fatwas that already have chunks (database mode, default: True)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_false',
        dest='skip_existing',
        help='Do not skip fatwas that already have chunks (database mode)'
    )
    parser.add_argument(
        '--reprocess-existing',
        action='store_true',
        help='Reprocess existing fatwas by deleting their chunks first (database mode)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of chunks to insert per batch (database mode, default: 100)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of fatwas to process (database mode)'
    )

    args = parser.parse_args()

    logger.info(f"Input mode: {'Database' if args.database_url else 'File'}")
    logger.info(f"Max tokens per chunk: {args.max_tokens}")

    # Initialize processor
    processor = Processor("aubmindlab/bert-base-arabertv02", args.max_tokens)

    # Process based on mode
    if args.database_url:
        # Database mode
        logger.info(f"Database URL: {args.database_url}")
        logger.info(f"Text column: {args.text_column}")
        logger.info(f"Skip existing: {args.skip_existing}")
        logger.info(f"Reprocess existing: {args.reprocess_existing}")

        # Initialize database manager
        db_manager = PostgreSQLManager(args.database_url)
        processor.db_manager = db_manager

        # Process database
        processor.process_database(
            text_column=args.text_column,
            skip_existing=args.skip_existing,
            delete_existing=args.reprocess_existing,
            batch_size=args.batch_size,
            limit=args.limit
        )
    else:
        # File mode (existing functionality)
        logger.info(f"Input folder: {args.input_dir}")
        logger.info(f"Output JSONL file: {args.output_file}")
        logger.info(f"Skipping header line: {args.skip_header}")

        processor.process_folder(args.input_dir, args.output_file, args.skip_header)

    logger.info("Processing complete!")
