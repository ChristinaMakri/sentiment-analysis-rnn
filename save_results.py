import pandas as pd

def save_results(results, filename='experiment_results.csv'):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
