'''
Author @ Lin Shi
Date: 2024/04/26

Reference Link: https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker/
'''

# Import necessary modules
# API Reference for SemanticChunker and OpenAIEmbeddings can be found at:
# SemanticChunker: https://api.python.langchain.com/en/latest/text_splitter/langchain_experimental.text_splitter.SemanticChunker.html
# OpenAIEmbeddings: https://api.python.langchain.com/en/latest/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


### Load example data
# Assume content.txt is a long document we can split up.

with open("dataset/GPT-generated_data_1.txt") as f:
    content = f.read()

# Create instances of SemanticChunker with different threshold types
percentile_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
)
standard_deviation_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation"
)
interquartile_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="interquartile"
)

# Split text using percentile method
percentile_docs = percentile_splitter.create_documents([content])
print("Percentile Method Example Output:")
print(percentile_docs[0].page_content)
print(f"Total Documents Split: {len(percentile_docs)}\n")

# Split text using standard deviation method
standard_deviation_docs = standard_deviation_splitter.create_documents([content])
print("Standard Deviation Method Example Output:")
print(standard_deviation_docs[0].page_content)
print(f"Total Documents Split: {len(standard_deviation_docs)}\n")

# Split text using interquartile method
interquartile_docs = interquartile_splitter.create_documents([content])
print("Interquartile Method Example Output:")
print(interquartile_docs[0].page_content)
print(f"Total Documents Split: {len(interquartile_docs)}\n")
