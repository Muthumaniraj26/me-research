import sys
import pandas as pd
import json
from mcp.server.fastmcp import FastMCP

# Initialize MCP Server
mcp = FastMCP("Solar_Research_Analyst")

@mcp.tool()
def get_performance_matrix():
    """Returns the 8-digit precision performance table for all 16 algorithms."""
    try:
        df = pd.read_csv("ultimate_high_precision_solar_results.csv")
        # Sorting by Test R2 so the AI sees the best models first
        df_sorted = df.sort_values(by="Test_Acc(R2)", ascending=False)
        return df_sorted.to_string(index=False)
    except Exception as e:
        return f"Error: Ensure the CSV file exists. {str(e)}"

@mcp.tool()
def analyze_model_comparison(metric: str = "Test_Acc(R2)"):
    """
    Analyzes which algorithm performed best for a specific metric.
    Allowed metrics: 'Train_Acc(R2)', 'Test_Acc(R2)', 'RMSE', 'MAE', 'MSE', 'Precision*'
    """
    df = pd.read_csv("ultimate_high_precision_solar_results.csv")
    if metric in ["RMSE", "MAE", "MSE"]:
        best_model = df.loc[df[metric].idxmin()]
    else:
        best_model = df.loc[df[metric].idxmax()]
    
    return f"The best model for {metric} is {best_model['Model']} with a value of {best_model[metric]:.8f}"

if __name__ == "__main__":
    mcp.run()