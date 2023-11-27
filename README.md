# Open Source RAG Demo using LangChain

Everything is run locally using LLaMa.cpp and LangChain Python wrappers.
This project is for demonstration purposes.

## Prepare

### Create directories

    mkdir sources models

### Download models

An example is to use LLaMa2 GGUF by TheBloke:

    wget -L https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf\?download\=true -O models/llama-2-7b-chat.Q4_K_M.gguf

### Sources

Add sources to the `sources` directory.
By default only PDF files are supported, but feel free to add functionality or change the loaders.

### Python environment

Create virtual python environment unless you like destroying your computer or know exactly why you don't need one.

    virtualenv .venv

or 

    python -m venv .venv

Activate the virtual environment

    # Windows
    .venv\Scripts\activate
    # macOS and Linux
    source .venv/bin/activate

### Install dependencies

    pip install -r requirements.txt

## Vectorizing sources

Once you have added the sources to the `sources` directory, store them in the vector store:

    python parse.py

This will persist the vector store in the directory `chroma_db`.
This way the vectorization does not need to be rerun unless you want to add more sources.

## Running

When the data is vectorized, you can start the assistant:

    python ai.py

Feel free to tweak parameters.
