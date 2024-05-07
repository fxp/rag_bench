import pandas as pd

def slice_data(input_csv: str, n: int, output_csv: str):
    """
    Slices the first `n` rows from the input CSV file and writes them to the output CSV file.

    Args:
        input_csv (str): Path to the input CSV file.
        n (int): Number of rows to slice.
        output_csv (str): Path to the output CSV file.
    """
    try:
        # Load the data from the CSV file
        df = pd.read_excel(input_csv)

        # Slice the first `n` rows
        sliced_df = df.head(n)

        # Write the sliced DataFrame to a new CSV file
        sliced_df.to_csv(output_csv, index=False)

        print(f"Sliced data successfully written to {output_csv}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__=="__main__":
    # Example usage
    slice_data('LLM_chunking/test_data/klg_base_100.xlsx', 10, 'LLM_chunking/test_data/klg_base_10.csv')
