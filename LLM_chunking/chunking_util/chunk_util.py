'''
Using LLMs for semantic chunking

Author @ Lin Shi
Github: Slimshilin
Date: 2024/05/01
'''
from .smp import load, dump
from functools import partial
import os
import os.path as osp
from tqdm import tqdm
import pandas as pd
from LLM_chunking.chat_api import OpenAIWrapper,ClaudeWrapper, GeminiWrapper
from LLM_chunking.prompts.system_prompt import build_prompt
import csv

def llm_semantic_chunk(data, llm, failure_count=5):
    if 'gpt' in llm:
        model = partial(OpenAIWrapper, llm, retry=10, timeout=150, verbose=True)()
    elif 'claude' in llm:
        model = partial(ClaudeWrapper, llm, retry=10, timeout=150, verbose=True)()
    elif 'gemini' in llm:
        model = partial(GeminiWrapper, llm, retry=20, timeout=150, verbose=True)()
        
    # Extracts the base name of the data file for later use in naming output directories and files.
    data_name = data.split('/')[-1].split('.')[0]

    # Defines the target name for output chunking results file
    target_name = f'chunking_output-{llm}.tsv'

    # Loads the data from the file.
    data = load(data)

    # Creates the directory for storing results.
    root = f'output/chunk_{data_name}_{llm}'
    os.makedirs(root, exist_ok=True)

    # Sets up a temporary file for intermediate results.
    tmp_file = osp.join(root, 'tmp.pkl')
    
    # Creates prompts.
    prompts = [build_prompt(data.iloc[i]) for i in range(len(data))]

    # print(prompts)

    # Maps comparison indices to their respective prompts.
    prompts_map = {x: p for x, p in zip(data.index, prompts)}

    ans = {}

    # Support resuming
    # Checks if a temporary file exists with some results and loads them.
    if osp.exists(tmp_file):
        ans = load(tmp_file)
        # Filters out any failed results.
        ans_ok = {x: y for x, y in ans.items() if 'Failed' not in y}

        # Removes failed results from the total count and updates the answers dictionary.
        if len(ans) != len(ans_ok):
            print(f'{len(ans) - len(ans_ok)} results removed during prefetching. ')
            ans = ans_ok

        # Updates the prompts map to exclude any prompts that already have answers.
        prompts_map = {x: y for x, y in prompts_map.items() if x not in ans}
    

    # Processes the prompts sequentially.
    for id, prompt in tqdm(list(zip(data.index, prompts))):
        if id in ans:
            continue
        ans[id] = model.generate(prompt)
        # TODO:
        # Currently the logic handles only stores the raw output of LLMs
        # Should be later employeed and adjusted using
        # parse_semantic_chunking_output()
        dump(ans, tmp_file)

    # Filters out any failed results after processing.
    ans_ok = {x: y for x, y in ans.items() if model.fail_msg not in y}
    if len(ans) != len(ans_ok):
        print(f'{len(ans) - len(ans_ok)} results failed. Rerun the command if you think that is too much. ')

    # Checks if the number of failed results is acceptable.
    if len(ans) - len(ans_ok) >= failure_count:
        print(f'{len(ans) - len(ans_ok)} results failed, which is more than {failure_count} records. Will not generate the inference result tsv. ')
        exit(-1)
    
    # Convert the dictionary to a DataFrame before saving, including the original column
    results = pd.DataFrame.from_dict(ans, orient='index', columns=[f'semantic_chunking_output-{llm}'])
    results.index.name = 'id'

    # Add the original content column from the `data` DataFrame
    results = results.join(data[['content_to_chunk']])

    # Reorder columns to ensure original content appears first
    results = results[['content_to_chunk', f'semantic_chunking_output-{llm}']]

    # Saves the combined data to a file using the `dump` function
    dump(results, f'{root}/{target_name}', quoting=csv.QUOTE_ALL)
    return results


def parse_semantic_chunking_output(semantic_chuking_output: str, original_complete_content: str) -> pd.DataFrame:
    '''
    Parse the semantic chunking output of LLM judges.
    Store the chunks' information for each chunk:
        - chunk_ID
        - original_content: Extract the original text according to the LLM output
        - semantic_description: The semantic description extracted from the LLM output
    '''
    
    pass
