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


## LLM-as-a-Judge evaluate chunking
Module in [here](./LLM-as-a-Judge_evaluation)

### API Config
If you haven't done so for LLM chunking, do the following API setup:


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


### Data Preparation
See [prepare_lc-LLM_data.py](./LLM-as-a-Judge_evaluation/data_preparation/prepare_lc-LLM_data.py) for detail. 

This prepares the `xlsx` or `csv` file (recommend `xlsx`) for LLM Judges to choose the better one between two chunking results - pairwise comparative assessment.

Store the data in [llm_judge/data](./LLM-as-a-Judge_evaluation/llm_judge/data) for later usage.


### LLM-as-a-Judge Evaluation
First go the the llm_judge folder:
```bash
cd LLM-as-a-Judge_evluation/llm_judge
```

Then run the following example script:
```bash
python3 subeval/subjective/sub_eval.py --data data/klg_base_10_chunked.csv --model lc-percentile lc-standard_deviation lc-interquartile gpt-3.5-turbo-1106 gpt-4-0613 --refm gpt-3.5-turbo-1106 --judge gpt-3.5-turbo-1106 --eval-nopt 2 --eval-proc 1 --mode dual
```

or 

```bash
chmod +x ./scripts/judge_klg_bas_10_3LC_2LLM.sh
./scripts/judge_klg_bas_10_3LC_2LLM.sh
```

Revise the scripts as needed.

**Argument Explanation:**
- `--data`: the relative path to your prepared data file. See [prepare_lc-LLM_data.py](./LLM-as-a-Judge_evaluation/data_preparation/prepare_lc-LLM_data.py) for detailed requirements. 
- `--model`: a list of models that are going to be evaluated, which should be included in your data file. These models will be all be compared to the `refm`, i.e., the reference/baseline model, that is also included in the data file.
- `--refm`: the reference or baseline model for comparison. It doesn't matter if you include the reference model in the --model list or not, because the program will discard exact matches.
- `--judge`: the judge model. Please follow the official use of APIs for judge names.
- `--eval-nopt`: option mode, i.e., the number of options the judge has.
    - 2: A is better than B; B is better than A
    - 3: A is better than B; B is better than A; Tie
- `--eval-proc`: for mutli-process usage. Recommend to set to 1 to avoid bugs.
- `--mode`: choose between `random` and `dual`. `dual` means to swap the order of the two responses in the query to LLM judges and evaluate twice (original + swapped). It tells if the judge model makes position bias (i.e., favors that response at a specific position rather than its content).