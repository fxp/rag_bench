import os
import pandas as pd

def prepare_lc_LLM_data(lc_output_files, LLM_output_files, save_to_path):
    # Initialize an empty list to store the concatenated data
    data = []

    # Process Langchain output files
    for file_path in lc_output_files:
        method = file_path.split("-")[-1].split(".")[0]  # Extract the method from the file name
        df_lc = pd.read_csv(file_path)
        lc_chunked_col = [col for col in df_lc.columns if col.startswith("lc_chunked_results")][0]
        df_lc = df_lc.rename(columns={lc_chunked_col: f"answer-lc-{method}"})
        data.append(df_lc)

    # Process LLM output files
    for file_path in LLM_output_files:
        df_LLM = pd.read_csv(file_path, sep="\t")
        semantic_chunking_col = [col for col in df_LLM.columns if col.startswith("semantic_chunking_output")][0]
        judge_name = semantic_chunking_col.replace("semantic_chunking_output-", "")  # Extract the judge name
        df_LLM = df_LLM.rename(columns={semantic_chunking_col: f"answer-{judge_name}"})
        data.append(df_LLM)

    # Merge the DataFrames based on the 'content_to_chunk' column
    merged_df = pd.merge(data[0], data[1], on="content_to_chunk")
    for i in range(2, len(data)):
        merged_df = pd.merge(merged_df, data[i], on="content_to_chunk")

    # Create the 'question' column
    merged_df["question"] = "Semantic chunking of the following content: " + merged_df["content_to_chunk"]

    # Create the 'index' column
    merged_df.reset_index(inplace=True)
    merged_df["index"] = merged_df.index.map(lambda x: f"question{x}")  # Use the format "question{index}"

    # Create empty 'evaluating_guidance' and 'reference_answer' columns
    merged_df["evaluating_guidance"] = ""
    merged_df["reference_answer"] = ""

    # Create the 'task' column with a fixed value
    merged_df["task"] = "semantic chunking"

    # Reorder the columns
    columns_order = ["question", "index", "evaluating_guidance", "task", "reference_answer"]
    columns_order.extend([col for col in merged_df.columns if col.startswith("answer-")])
    merged_df = merged_df[columns_order]

    # Save the resulting DataFrame to the specified path
    os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
    merged_df.to_csv(save_to_path, index=False)


    print(f"Merged results saved to {save_to_path}")

    return merged_df


if __name__=="__main__":
    lc_output_files = ["output/chunk_klg_base_10_lc/chunking_output-lc-percentile.csv",
                       "output/chunk_klg_base_10_lc/chunking_output-lc-standard_deviation.csv",
                       "output/chunk_klg_base_10_lc/chunking_output-lc-interquartile.csv"]
    LLM_output_files = ["output/chunk_klg_base_10_gpt-3.5-turbo-1106/chunking_output-gpt-3.5-turbo-1106.tsv",
                        "output/chunk_klg_base_10_gpt-4-0613/chunking_output-gpt-4-0613.tsv"]
    save_to_path = "LLM-as-a-Judge_evaluation/llm_judge/data/klg_base_10_chunked.csv"

    prepare_lc_LLM_data(lc_output_files=lc_output_files,
                        LLM_output_files=LLM_output_files,
                        save_to_path=save_to_path)