import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error in load_data: {e}")
        return None
