# LLM-TOOL-KIT

## Description

This project focuses on Natural Language Processing (NLP) for Arabic text. It utilizes transformer models and other NLP techniques for tasks such as data processing and semantic analysis. The project uses MongoDB for data storage and includes utilities for logging and potentially reward functions for reinforcement learning or custom training loops.

## Features

*   Semantic analysis
*   Training Embedding Model
*   Training Tokenizer
*   Finetuning Reasoning Model
*   Finetuning SFT Models
*   Data processing pipelines
*   Integration with MongoDB
*   Customizable logging
*   Potential for reinforcement learning or custom model training

## Installation

1.  **Clone the repository:**
    ```bash
    git clone 'https://github.com/ghyathmoussa/llm-tool-kit.git'
    cd `llm-tool-kit`
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    Create a `.env` file in the project root and add the following variables:
    ```env
    MONGO_URI=<your_mongodb_uri>
    MONGO_DB_NAME=<your_mongodb_database_name>
    MONGO_COLLECTION_NAME=<your_mongodb_collection_name>
    ```

## Usage

* Tokenizer Model

```
python3 tokenizer_model.py \
    --tokenizer-name "custom-arabic-tokenizer" \
    --vocab-size 32000 \
    --max-length 4096 \
    --model-path "/path/to/tokenizer.json" \
    --texts-source "data1.jsonl" \
    --min-frequency 2 \
    --batch-size 1000 \
    --max-samples 10000 \
    --special-tokens "[PAD]" "[UNK]" "[CLS]" "[SEP]" "[MASK]"
```

Refer to the individual notebooks for specific instructions and execution details.

## Configuration

The project configuration is managed in `config.py`. This file loads environment variables from a `.env` file located in the project root. Key configuration variables include:

*   `MONGO_URI`: MongoDB connection string.
*   `MONGO_DB_NAME`: Name of the MongoDB database.
*   `MONGO_COLLECTION_NAME`: Name of the MongoDB collection.

Ensure these are correctly set in your `.env` file before running the project.

## Data

The project appears to use the [`Tashkeela-arabic-diacritized-text-utf8-0.3`](https://sourceforge.net/projects/tashkeela/files/Tashkeela-arabic-diacritized-text-utf8-0.3.zip/download) dataset, which can be found in the `Tashkeela-arabic-diacritized-text-utf8-0.3/` directory. Processed data or data used for modeling may be stored in json file.

The `source_data/` directory might contain raw or initial datasets, and the `data/` directory could be used for processed or intermediate data files.



## Docker

A `Dockerfile` is provided for containerizing the application. To build and run the Docker image:

1.  **Build the image:**
    ```bash
    docker build -t nlp-project .
    ```
2.  **Run the container:**
    ```bash
    docker run -d --env-file .env -p <host_port>:<container_port> nlp-project
    ```
    (You might need to adjust port mappings and ensure the container has access to necessary resources like MongoDB if it's not running within the same Docker network).

## Project Structure

```
.
├── Dockerfile
├── LICENSE
├── README.md
├── config.py
├── data/
├── evals/
├── helpers/
├── logs/
├── models/
├── notebooks/
│   ├── process_data.ipynb
│   └── semantic_process.ipynb
├── outputs/
├── requirements.txt
├── source_data/
├── Tashkeela-arabic-diacritized-text-utf8-0.3/
└── utils/
    ├── __init__.py
    ├── logger.py
    └── reward_functions.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to the project's coding standards and includes tests where appropriate.

## License

This project is licensed under the terms of the LICENSE file.