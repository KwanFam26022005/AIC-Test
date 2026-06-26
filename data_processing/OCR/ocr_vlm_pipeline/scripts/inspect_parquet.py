import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Inspect pipeline parquet files.")
    parser.add_argument("file", help="Path to the Parquet file to inspect (e.g., outputs/L25_V001/ppocr_raw.parquet)")
    parser.add_argument("--head", type=int, default=10, help="Number of rows to display")
    parser.add_argument("--cols", nargs="*", help="Specific columns to display")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: File {path} does not exist.")
        return

    try:
        df = pd.read_parquet(path)
        print(f"\n--- File: {path.name} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nPreview:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        display_df = df
        if args.cols:
            valid_cols = [c for c in args.cols if c in df.columns]
            if valid_cols:
                display_df = df[valid_cols]
        
        print(display_df.head(args.head))
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
