''' Langchain Semantic Chunker, chunking a df with column `content_to_chunk`
    Author @ Lin Shi
    Date: 2024/05/10
'''
import os
import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

def lc_semantic_chunking(df, method="percentile", save_to_file=None):
    # Create an instance of SemanticChunker with the specified method
    if method == "percentile":
        text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")
    elif method == "standard_deviation":
        text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation")
    elif method == "interquartile":
        text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="interquartile")
    else:
        raise ValueError("Invalid chunking method. Choose from 'percentile', 'standard_deviation', or 'interquartile'.")

    # Apply semantic chunking to each row of the DataFrame
    df[f"lc_chunked_results-{method}"] = df["content_to_chunk"].apply(lambda content: 
        "\n".join([f"{i+1}:::{doc.page_content}" for i, doc in enumerate(text_splitter.create_documents([content]))]))

    # Save the resulting DataFrame to a file if specified
    if save_to_file:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
        df.to_csv(save_to_file, index=False)

    return df

if __name__ == "__main__":
    df = pd.read_csv('LLM_chunking/test_data/klg_base_10.csv')
    lc_semantic_chunking(df, method="percentile", save_to_file="output/chunk_klg_base_10_lc/chunking_output-lc-percentile.csv")
    lc_semantic_chunking(df, method="standard_deviation", save_to_file="output/chunk_klg_base_10_lc/chunking_output-lc-standard_deviation.csv")
    lc_semantic_chunking(df, method="interquartile", save_to_file="output/chunk_klg_base_10_lc/chunking_output-lc-interquartile.csv")