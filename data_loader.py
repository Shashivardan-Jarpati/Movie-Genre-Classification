import pandas as pd

def load_data(file_path="dataset.csv"):
    data = pd.read_csv(file_path)

    # Input (movie plot) and output (genre)
    X = data["plot"]
    y = data["genre"]

    return X, y
