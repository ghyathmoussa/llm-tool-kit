from datasets import Dataset, DatasetDict
from huggingface_hub import login
from utils.logger import logger
import argparse
import pandas as pd
from typing import Union

# TODO: Ensure you are logged in to Hugging Face before calling the upload function.
# You can do this by running `huggingface-cli login` in your terminal
# or by calling `login()` here, which might prompt for a token.
# Example:
# login(token="YOUR_HF_TOKEN") # or login() to be prompted

def upload_to_huggingface(
    dataset_obj: Union[Dataset, DatasetDict],
    repo_id: str,
    private: bool = False,
    token: Union[str, bool, None] = None
) -> None:
    
    """
    Uploads a datasets.Dataset or datasets.DatasetDict object to the Hugging Face Hub.

    Args:
        dataset_obj: The Dataset or DatasetDict object to upload.
        repo_id: The ID of the repository on Hugging Face.
                 This should be in the format "username/dataset_name"
                 or "organization_name/dataset_name".
        private: Whether the dataset should be private. Defaults to False.
        token: Your Hugging Face API token. If not provided, and you are not logged in
               via `huggingface-cli`, the function might try to use a cached token
               or prompt for login if `huggingface_hub` is configured to do so.
               It can be a string (token) or True (to use cached token).
    """
    if not isinstance(dataset_obj, (Dataset, DatasetDict)):
        raise TypeError(
            "dataset_obj must be an instance of datasets.Dataset or datasets.DatasetDict"
        )
    if not repo_id or len(repo_id.split('/')) != 2:
        raise ValueError(
            "repo_id must be in the format 'username/dataset_name' or 'org/dataset_name'"
        )

    logger.info(f"Uploading dataset to Hugging Face repository: {repo_id}")
    try:
        dataset_obj.push_to_hub(repo_id, private=private, token=token)
        logger.info(f"Successfully uploaded dataset to {repo_id}")
        logger.info(f"Access it at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        logger.error(f"An error occurred during upload: {e}")
        raise

if __name__ == '__main__':
    """
    To run this script:
    1. Ensure you have `datasets`, `huggingface_hub`, and `pandas` installed.
    2. If not providing a token, ensure you are logged in via `huggingface-cli login`.
    3. Execute the script with required arguments, for example:
    4. `python utils/upload_to_hugginface.py --repo_id YOUR_USERNAME/YOUR_DATASET_NAME --data_path path/to/your/data.csv`
    5. To make it private: `... --private`
    6. To use a token: `... --token YOUR_HF_TOKEN`
    """
    parser = argparse.ArgumentParser(description="Upload a dataset to Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face (e.g., 'username/dataset_name')."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the CSV data file to upload."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Set the dataset to private on Hugging Face. Default is public."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token. If not provided, behavior depends on cached credentials or global login."
    )

    args = parser.parse_args()

    try:
        login(token=args.token)
        logger.info(f"Loading dataset from CSV file: {args.data_path}")
        df = pd.read_json(args.data_path, lines=True)
        example_dataset = Dataset.from_pandas(df)

        logger.info(f"Attempting to upload dataset from '{args.data_path}' to Hugging Face repository: {args.repo_id}")

        upload_to_huggingface(
            dataset_obj=example_dataset,
            repo_id=args.repo_id,
            private=args.private,
            token=args.token
        )


    except ImportError:
        logger.error("Pandas library is not installed. Required for loading CSV data in this example.")
        logger.error("Please install it using: pip install pandas datasets huggingface_hub")
    except FileNotFoundError:
        logger.error(f"Data file not found at: {args.data_path}")
    except Exception as e:
        logger.error(f"An error occurred in the main execution block: {e}")
        raise
