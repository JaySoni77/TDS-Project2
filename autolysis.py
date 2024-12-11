import argparse

import openai
import os
import requests

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Loading CSV
parser = argparse.ArgumentParser(description="Automated dataset analysis.")
parser.add_argument("csv_file", type=str, help="Path to the CSV dataset.")
args = parser.parse_args()
csv_file = args.csv_file

#create folder with the name of CSV
folder_name = os.path.splitext(csv_file)[0]
os.makedirs(folder_name, exist_ok=True)
# print(type(csv_file))

#Checking encoding and creating dataframe
import chardet

with open(csv_file, 'rb') as f:
    result = chardet.detect(f.read())
# print(result)
encoding = result['encoding']

data = pd.read_csv(csv_file, encoding=encoding)

#Generating Statestical Analysis
numeric_data = data.select_dtypes(include=['number'])
summary = numeric_data.describe()
missing_values = data.isnull().sum()
correlation_matrix = numeric_data.corr()

#Create chart
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix Heatmap", fontsize=12)
heatmap_path = os.path.join(folder_name, "correlation_matrix.png")
# plt.figure(figsize=(5, 5))  # Set figure size for 512x512 px
plt.tight_layout()  # Adjust layout to prevent clipping
# plt.savefig(heatmap_path, dpi=100)  # DPI of 100 ensures 512x512 px
plt.savefig(heatmap_path, dpi=100, bbox_inches='tight')
plt.close()

#AI Prompt
prompt = f"""
I have a dataset with the following columns:
{list(data.columns)}

The summary statistics are:
{summary}

The Missing values are:
{missing_values}

And the Correlation matrix is:
{correlation_matrix}

Provide Potential insight from above information and write a story about your analysis.
Describe:
1 - The data you received, briefly
2 - The analysis you carried out
3 - The insights you discovered
4 - The implications of your findings (i.e. what to do with the insights)
"""


# key = openai.api_key = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZHMzMDAwMDgxQGRzLnN0dWR5LmlpdG0uYWMuaW4ifQ.smhUC1nO2VQZFaD8VnPMjEmT9GjhJJYknCFry4Rs0YI"
key = os.environ["AIPROXY_TOKEN"]

#Function to generate analysis and insight from AI
def ask_llm(prompt):
    # API endpoint
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    # Request headers
    headers = {"Content-Type": "application/json","Authorization": f"Bearer {key}"}

    # Request data
    data = {
        "model": "gpt-4o-mini",  
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt 
            }
        ]
    }
    # Make the API call
    response = requests.post(url, headers=headers, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        # print(result)  # Print the response from OpenAI
        # print(result['choices'][0]['message']['content']) # Print the response from OpenAI
        res = result['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code}, {response.text}")
        res = [response.status_code, response.text]
    return res

insights = ask_llm(prompt)


readme_path = os.path.join(folder_name, "README.md")
with open(readme_path, "w") as file:
    file.write("# Automated Analysis Report\n")
    file.write(f"# Dataset: {csv_file}\n\n")
    file.write("### Key Insights\n")
    file.write(insights + "\n")
    file.write("### Visualizations\n")
    file.write("![Correlation Matrix](correlation_matrix.png)\n")