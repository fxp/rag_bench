# rag_bench

## Langchain-text-splitters
First install text splitter
```bash
pip install -qU langchain-text-splitters
```
### Semantic Chunking
```bash
pip install --quiet langchain_experimental langchain_openai
```



## LLM Chunking

### API Config
Store your API key in a json file, for example, `keys.json`.

```json
{
    "openai-keys":[],
    "claude-keys":[],
    "gemini-keys":[]
}
```

And then:

```bash
export KEYS="path/to/keys.json"
```

### Data preparation

Prepare your data in a `xlsx` or `csv` format, such that there exists a column `content_to_chunk`. The LLM chunker will ONLY chunk that column.

The example data or test data are stored in [LLM_chunking/test_data](./LLM_chunking/test_data).,

### Example Usage
First, export Python path from root:
```bash
export PYTHONPATH=$PWD
```

Run the following script from root:
```bash
python LLM_chunking/main_LLM_chunking.py --data relative/path/to/data/file --llm llm/for/semantic/chunking
```

For example:
```bash
python LLM_chunking/main_LLM_chunking.py --data LLM_chunking/test_data/klg_base_10.csv --llm gpt-3.5-turbo-1106
```

If you may also run scripts from [LLM_chunking/scripts](./LLM_chunking/scripts).

For example:
```bash
chmod +x ./LLM_chunking/scripts/run_gpt-3.5-turbo-1106_klg_base_10.sh
```
```bash
./LLM_chunking/scripts/run_gpt-3.5-turbo-1106_klg_base_10.sh
```

### Chunking output
The LLM semantic chunking output will be stored in [output](./output) as a `tsv` file.

The `tmp.pkl` is the temporary file for intermediate chunking outputs. Therefore, if the chunking process is interrupted, it will read this temporary file and keep running without overloading the previous results when the same script is run the next time.

If you want to chunk the same data file with the same judge multiple times, be sure to **rename** the output directory or file name to avoid any conflicting issue.


### Program Functionality
[chat_api](./LLM_chunking/chat_api) utilizes API Wrappers (OpenAIWrapper, ClaudeWrapper, etc.) for API calls.

[prompts](./LLM_chunking/prompts/system_prompt.py) provides a function to build Chinese/English prompts for semantic chunking.

[chunk_util.py](./LLM_chunking/chunking_util/chunk_util.py) is the semantic chunking function that handles the main chunking logic

[smp.py](./LLM_chunking/chunking_util/smp.py) provides helper functions for loading and dumping data files.

[main_LLM_chunking.py](./LLM_chunking/main_LLM_chunking.py) is the main function call for LLM semantic chunking.