'''
Main function call: using LLMs for semantic chunking

Author @ Lin Shi
Github: Slimshilin
Date: 2024/05/07
'''

import argparse
import pandas as pd
from LLM_chunking.chunking_util.smp import load, dump, timestr
from LLM_chunking.chunking_util.chunk_util import llm_semantic_chunk

def parse_args():
    parser = argparse.ArgumentParser(
        description="Conduct semantic chunking using LLMs "
    )
    parser.add_argument(
        "--data", 
        type=str, 
        help=("An excel/csv/tsv with a column 'content_to_chunk'."))
    parser.add_argument("--llm", type=str, default='gpt-4-1106-preview',help="The LLM for semantic chunking")    
    args = parser.parse_args()
    return args


### Main Logic for LLM semantic chunking ###
def LLM_semantic_chunking_call():
    args = parse_args()
    
    chunking_results_file = llm_semantic_chunk(args.data, llm=args.llm)

    print(chunking_results_file)

    
if __name__ == '__main__':
    LLM_semantic_chunking_call()