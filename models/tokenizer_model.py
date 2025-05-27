from config import PROJECT_ROOT
from typing import Dict, List, Generator, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from random import seed
from load_data import get_data_iter
import logging
from pathlib import Path
import gc  # For garbage collection
import json
import argparse

seed(42)

class TokenizerModel:
    def __init__(self, tokenizer_name: str, vocab_size: int, max_length: int, model_path: str = None):
        self.tokenizer_name = tokenizer_name
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        
        # Load the tokenizer if model_path exists, otherwise initialize a new one
        if model_path and Path(model_path).exists():
            self.tokenizer = Tokenizer.from_file(model_path)
            self.tokenizer.enable_padding(length=self.max_length)
            self.tokenizer.enable_truncation(max_length=self.max_length)
            logging.info(f"Loaded tokenizer from {model_path}")
        else:
            self.tokenizer = Tokenizer(BPE())
            self.model_path = model_path if model_path else f"{PROJECT_ROOT}/outputs/custom_tokenizer.json"
            logging.info("Initialized a new tokenizer.")

    def _prepare_text(self, text):
        # Simple text preparation, can be expanded
        return text.lower()

    def encode_batch(self, texts: List[str], batch_size: Optional[int] = None):
        """
        Encodes a batch of texts, optionally processing in smaller batches.
        """
        if not batch_size or len(texts) <= batch_size:
            # Process all at once if batch_size not specified or texts fit in one batch
            prepared_texts = [self._prepare_text(text) for text in texts]
            return self.tokenizer.encode_batch(prepared_texts)
        else:
            # Process in smaller batches
            all_encodings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                prepared_batch = [self._prepare_text(text) for text in batch]
                encodings = self.tokenizer.encode_batch(prepared_batch)
                all_encodings.extend(encodings)
                gc.collect()  # Encourage garbage collection
            return all_encodings

    def train(self,
              min_frequency: int,
              special_tokens: List[str],
              texts_source: str,  # Path to data file
              batch_size: int = 1000,  # Process in batches
              max_samples: Optional[int] = None  # Limit samples for debugging
             ):
        """
        Memory-efficient training for large datasets.
        """
        # Set up the trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )

        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Set truncation and padding
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(
            pad_id=special_tokens.index("[PAD]") if "[PAD]" in special_tokens else 0, 
            length=self.max_length
        )
        
        # Use a generator that yields one text at a time to save memory
        def text_generator():            
            data_iter = get_data_iter(path = texts_source)  # Get one text at a time
            count = 0
            for text in data_iter:
                yield self._prepare_text(text)
                count += 1
                if max_samples and count >= max_samples:
                    break
                if count % 10000 == 0:  # Log progress
                    logging.info(f"Processed {count} texts for training")
        
        # Train from the generator
        self.tokenizer.train_from_iterator(text_generator(), trainer)
        
        # Save the tokenizer
        save_path = Path(self.model_path)
        # Ensure the directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(save_path))
        logging.info(f"Tokenizer trained and saved to {save_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train and use a custom tokenizer.")
    parser.add_argument("--tokenizer-name", type=str, required=True, help="Name of the tokenizer.")
    parser.add_argument("--vocab-size", type=int, required=True, help="Vocabulary size for the tokenizer.")
    parser.add_argument("--max-length", type=int, required=True, help="Maximum length of tokenized sequences.")
    parser.add_argument("--model-path", type=str, required=False, help="Path to an existing tokenizer model file.")
    parser.add_argument("--texts-source", type=str, required=False, help="Path to the text data file for training.")
    parser.add_argument("--min-frequency", type=int, required=False, default=2, help="Minimum frequency for subword units.")
    parser.add_argument("--batch-size", type=int, required=False, default=1000, help="Batch size for training.")
    parser.add_argument("--max-samples", type=int, required=False, help="Maximum number of samples for training.")
    parser.add_argument("--special-tokens", type=str, nargs="+", required=False, 
                        default=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], 
                        help="List of special tokens.")

    args = parser.parse_args()

    tokenizer_path = Path(args.model_path) if args.model_path else Path(f"{PROJECT_ROOT}/outputs/custom_tokenizer.json")

    tokenizer = TokenizerModel(
        tokenizer_name=args.tokenizer_name,
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        model_path=str(tokenizer_path)
    )

    # Only train if the tokenizer file doesn't exist
    if not tokenizer_path.exists() and args.texts_source:
        logging.info("Tokenizer file not found. Starting training...")
        tokenizer.train(
            min_frequency=args.min_frequency,
            special_tokens=args.special_tokens,
            texts_source=args.texts_source,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
    else:
        logging.info(f"Using existing tokenizer found at {tokenizer_path}")

    # Example encoding with batching for large datasets
    sample_texts = [
        "هذا مثال للنص الأول.",
        "وهذا مثال آخر أطول قليلاً لاختبار الترميز.",
        "جملة قصيرة."
    ]

    encoded_batch = tokenizer.encode_batch(sample_texts)

    logging.info(f"Encoded batch ({len(encoded_batch)} texts):")
    for i, encoding in enumerate(encoded_batch):
        logging.info(f"Text {i+1}:")
        logging.info(f" Tokens: {encoding.tokens}")
        logging.info(f" IDs: {encoding.ids}")
        logging.info(f" Attention Mask: {encoding.attention_mask}")
