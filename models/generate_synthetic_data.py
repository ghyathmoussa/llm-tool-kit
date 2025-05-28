import json
import os
import argparse
import openai # Added for OpenRouter
from helpers.get_prompt import get_prompt
from config import API_KEY  # Assuming you have a config.py with API_KEY defined

def generate_qa_from_text_with_llm(text_content: str, num_qa_pairs: int = 3, api_key: str = None, llm_model: str = "openai/gpt-4o-mini"):
    """
    Generates question-answer pairs from the given Arabic text using an LLM via OpenRouter.

    Args:
        text_content (str): The Arabic text from which to generate Q&A.
        num_qa_pairs (int): The desired number of Q&A pairs.
        api_key (str, optional): The OpenRouter API key. Defaults to None.
        llm_model (str, optional): The model to use via OpenRouter. Defaults to "openai/gpt-4o-mini".

    Returns:
        list: A list of dictionaries, where each dictionary is
              {"question": "...", "answer": "..."}.
              Returns an empty list if generation fails or no content.
    """
    if not text_content:
        return []

    if not api_key:
        api_key = os.environ.get("API_KEY")
        if not api_key:
            print("Error: OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable or use --llm-api-key.")
            return []

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt_text = get_prompt(language='arabic', num_qa_pairs=num_qa_pairs, text_content=text_content)

    print(f"DEBUG: Calling OpenRouter with model {llm_model} for text starting with: {text_content[:100]}...")

    try:
        completion = client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                },
            ]
        )
        
        response_content = completion.choices[0].message.content
        print(f"DEBUG: Raw LLM response: {response_content}") # For debugging

        try:
            qa_pairs = json.loads(response_content)
            if not isinstance(qa_pairs, list): # Ensure it's a list
                 print(f"Warning: LLM response was not a list of Q&A pairs. Response: {response_content}")
                 return []
            return qa_pairs
        except json.JSONDecodeError:
            print(f"Error: Could not decode LLM response as JSON. Response: {response_content}")
            # Attempt to extract Q&A pairs using a fallback mechanism if JSON parsing fails
            # This is a placeholder for a more robust parsing strategy
            # For now, we'll return an empty list if JSON parsing fails
            # A more advanced approach might involve regex or other parsing techniques
            # based on the expected (but not strictly JSON) output format.
            # Example (very basic, needs refinement):
            # qa_list = []
            # if "question:" in response_content.lower() and "answer:" in response_content.lower():
            #     # Simple split logic, highly dependent on LLM's non-JSON output format
            #     # This part needs to be carefully designed based on observed LLM outputs
            #     pass 
            return []

    except openai.APIError as e:
        print(f"OpenRouter API error: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during LLM call: {e}")
        return []


def create_synthetic_data(input_file_path, output_file_path, qa_per_chunk, api_key=None):
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

                # Generate Q&A pairs using the OpenRouter LLM function
                qa_pairs = generate_qa_from_text_with_llm(
                    text_content=original_text,
                    num_qa_pairs=qa_per_chunk,
                    api_key=api_key # Pass the API key
                )

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
    # Default OpenRouter model (example)
    DEFAULT_LLM_MODEL = "openai/gpt-4o-mini" # You can change this
    
    parser = argparse.ArgumentParser(description="Generate synthetic Q&A data for fine-tuning.")
    parser.add_argument("--input-file", type=str, required=True, default=INPUT_FILE_PATH, help="Path to the input JSONL file.")
    parser.add_argument("--output-file", type=str, required=True, default=OUTPUT_FILE_PATH, help="Path to the output JSONL file.")
    parser.add_argument("--qa-per-chunk", type=int, default=QA_PAIRS_PER_CHUNK, help="Number of Q&A pairs to generate per text chunk.")
    parser.add_argument("--llm-api-key", type=str, default=API_KEY, help="API key for the LLM service (e.g., OpenRouter). If not provided, tries to use OPENROUTER_API_KEY env var.")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL, help=f"The LLM model to use via OpenRouter (default: {DEFAULT_LLM_MODEL}).")

    args = parser.parse_args()
    
    create_synthetic_data(args.input_file, args.output_file, args.qa_per_chunk, api_key=args.llm_api_key)