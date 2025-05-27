import json
import os
import argparse
from helpers.get_prompt import get_prompt

def generate_qa_from_text_with_llm(prompt: str, text_content: str, num_qa_pairs: int = 3):
    """
    Generates question-answer pairs from the given Arabic text using an LLM.

    Args:
        text_content (str): The Arabic text from which to generate Q&A.
        num_qa_pairs (int): The desired number of Q&A pairs.

    Returns:
        list: A list of dictionaries, where each dictionary is
              {"question": "...", "answer": "..."}.
              Returns an empty list if generation fails or no content.
    """
    if not text_content:
        return []

    # --- !!! IMPORTANT: LLM Integration Point !!! ---
    # You need to replace the following placeholder logic with actual calls
    # to an LLM API (e.g., OpenAI, Cohere, a self-hosted model via Hugging Face).
    # Ensure the LLM is proficient in Arabic and can follow instructions for Q&A generation.

    prompt = get_prompt(language='arabic', num_qa_pairs=num_qa_pairs, text_content=text_content)
    
    print(f"DEBUG: Simulating LLM call for text starting with: {text_content[:100]}...") # For debugging

    # ---- Placeholder LLM Call Simulation ----
    # Replace this section with your actual LLM API call and response parsing.
    # Example using a hypothetical `call_my_arabic_llm` function:
    # try:
    #     llm_response_str = call_my_arabic_llm(prompt)
    #     generated_pairs = json.loads(llm_response_str) # Assuming LLM returns JSON string
    #     if isinstance(generated_pairs, list) and all(isinstance(p, dict) and "question" in p and "answer" in p for p in generated_pairs):
    #         return generated_pairs[:num_qa_pairs] # Ensure we don't exceed num_qa_pairs
    #     else:
    #         print(f"Warning: LLM response format is not as expected for text: {text_content[:50]}...")
    #         return []
    # except Exception as e:
    #     print(f"Error calling LLM or parsing response: {e}")
    #     return []
    # ---- End Placeholder LLM Call Simulation ----

    # Dummy placeholder output for demonstration if no LLM is integrated:
    # To make this script runnable without an LLM for testing purposes,
    # we'll generate very simple dummy data.
    # REMOVE OR REPLACE THIS WHEN YOU INTEGRATE A REAL LLM.
    dummy_pairs = []
    for i in range(1, num_qa_pairs + 1):
        dummy_pairs.append({
            "question": f"ما هو السؤال النموذجي رقم {i} المستنبط من هذا النص؟ (مثال)",
            "answer": f"هذه هي الإجابة النموذجية رقم {i} المستنبطة من النص. (مثال)"
        })
    return dummy_pairs


def create_synthetic_data(input_file_path, output_file_path, qa_per_chunk):
    """
    Reads data from input_file_path, generates synthetic Q&A pairs,
    and writes them to output_file_path in an instruction-following format.
    """
    print(f"Starting synthetic data generation from '{input_file_path}'...")
    count_processed_lines = 0
    count_generated_qa_pairs = 0

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line_number, line in enumerate(infile, 1):
            try:
                data_entry = json.loads(line)
                original_text = data_entry.get("text")

                if not original_text:
                    print(f"Warning: Skipping line {line_number} due to missing 'text' field.")
                    continue

                # Generate Q&A pairs using the placeholder (or your actual LLM function)
                # Pass the original text and the number of Q&A pairs desired
                qa_pairs = generate_qa_from_text_with_llm(original_text, num_qa_pairs=qa_per_chunk)

                if not qa_pairs:
                    print(f"Warning: No Q&A pairs generated for line {line_number} (text starting: '{original_text[:50]}...').")
                    continue

                for qa in qa_pairs:
                    if not (isinstance(qa, dict) and "question" in qa and "answer" in qa):
                        print(f"Warning: Skipping malformed Q&A pair on line {line_number}: {qa}")
                        continue
                    
                    # Format for instruction fine-tuning
                    instruction_data = {
                        "instruction": qa["question"],
                        "input": "",  # Input can be empty if instruction is self-contained
                                      # Or, you could put data_entry.get("chunk_id", "") here as a reference
                        "output": qa["answer"],
                        "source_document_info": { # Optional: for traceability
                            "original_source": data_entry.get("source_document"),
                            "original_chunk_id": data_entry.get("chunk_id"),
                            "original_text_preview": original_text[:200] + "..." # Preview for easier checking
                        }
                    }
                    outfile.write(json.dumps(instruction_data, ensure_ascii=False) + '\n')
                    count_generated_qa_pairs += 1
                
                count_processed_lines += 1
                if count_processed_lines % 50 == 0: # Log progress every 50 source lines
                    print(f"Processed {count_processed_lines} lines from input, generated {count_generated_qa_pairs} Q&A pairs so far...")

            except json.JSONDecodeError:
                print(f"Warning: Skipping line {line_number} due to JSON decode error: {line.strip()}")
            except Exception as e:
                print(f"Error processing line {line_number}: {line.strip()}. Error: {e}")

    print(f"Synthetic data generation complete.")
    print(f"Processed {count_processed_lines} lines from the input file.")
    print(f"Generated a total of {count_generated_qa_pairs} Q&A pairs.")
    print(f"Output written to '{output_file_path}'")


if __name__ == "__main__":
    # --- Configuration ---
    # It's good practice to define paths at the top or get them from arguments/env vars
    INPUT_FILE_PATH = os.path.join('data', 'processed_data.jsonl')
    OUTPUT_FILE_PATH = os.path.join('data', 'synthetic_finetuning_data.jsonl')
    # Number of Q&A pairs to try and generate per text chunk
    # Adjust this based on your needs and LLM capabilities
    QA_PAIRS_PER_CHUNK = 3
    
    parser = argparse.ArgumentParser(description="Generate synthetic Q&A data for fine-tuning.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSONL file.", default=INPUT_FILE_PATH)
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output JSONL file.", default=OUTPUT_FILE_PATH)
    parser.add_argument("--qa-per-chunk", type=int, default=3, help="Number of Q&A pairs to generate per text chunk.", default=QA_PAIRS_PER_CHUNK)
    parser.add_argument("--llm-api-key", type=str, help="API key for the LLM service (if applicable).")

    args = parser.parse_args()

    create_synthetic_data(args.input_file, args.output_file, args.qa_per_chunk)

    # Example of how you might fine-tune (conceptual):
    # from transformers import Trainer, TrainingArguments
    # train_dataset = load_dataset("json", data_files=OUTPUT_FILE_PATH, split="train")
    # model = AutoModelForCausalLM.from_pretrained("your-base-arabic-model")
    # tokenizer = AutoTokenizer.from_pretrained("your-base-arabic-model")
    # ... setup tokenizer, data collator, training arguments ...
    # trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, ...)
    # trainer.train()