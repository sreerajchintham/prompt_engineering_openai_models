# Prompt Engineering for OpenAI Models

This Google Colab notebook provides a comprehensive framework for experimenting with and evaluating various prompt engineering techniques when interacting with OpenAI's Large Language Models (LLMs) for a multiple-choice question-answering task.

## Overview

The notebook automates the process of testing different prompting strategies (Zero-shot, Few-shot, K-NN Few-shot, Chain-of-Thought, and Self-Consistency) across multiple OpenAI models (GPT-3.5, GPT-4, GPT-4o). It evaluates their performance on a subset of a medical question-answering dataset, provides robust API interaction with retry mechanisms, and visualizes the accuracy results for easy comparison.

## Key Features and Functionality

*   **Diverse Prompting Strategies:** Implements and compares common prompt engineering techniques:
    *   **Zero-shot:** Direct question answering without examples.
    *   **Few-shot:** Provides a small set of example question-answer pairs.
    *   **K-NN Few-shot:** Dynamically selects the most relevant examples from the training data using TF-IDF for few-shot prompting.
    *   **Chain-of-Thought (CoT):** Prompts the model to think step-by-step before answering.
    *   **Self-Consistency:** Generates multiple CoT answers and selects the majority vote.
*   **OpenAI Model Integration:** Seamlessly interacts with `gpt-3.5-turbo`, `gpt-4-turbo` (aliased as "GPT-4.1"), and `gpt-4o` models via the OpenAI API.
*   **Robust API Calling:** Includes a custom function with retry logic to handle `RateLimitError`, `Timeout`, `APIError`, and `AuthenticationError`.
*   **Automated Evaluation:** Iterates through specified models and techniques, processes test questions, and records predictions against actual answers.
*   **Results Persistence:** Saves the detailed evaluation results to a CSV file (`medqa_eval_results.csv`) after each model-technique combination.
*   **Performance Summary & Visualization:** Calculates and displays accuracy summaries for all tested combinations and generates a clear bar chart for visual analysis.

## Technologies/Libraries Used

*   **Python 3**
*   **`openai`**: For interacting with OpenAI's API.
*   **`pandas`**: For data manipulation and analysis of results.
*   **`scikit-learn`**: Specifically `TfidfVectorizer` and `cosine_similarity` for K-NN example selection.
*   **`matplotlib`**: For visualizing the accuracy results.
*   **`json`**: For loading dataset files.
*   **`time`**: For implementing delays in API calls (e.g., retries).
*   **`tqdm`**: For displaying progress bars during evaluations.
*   **`google.colab.userdata`**: For securely managing API keys in Colab.

## Main Sections and Steps

The notebook is structured into two main code cells:

### Cell 1: Setup, Data Loading, Model Interaction, and Evaluation

1.  **Environment Setup:** Imports necessary libraries and configures the OpenAI client using an API key stored in Colab secrets.
2.  **Data Loading:** Loads `train(1).jsonl` and `test(1).jsonl` datasets, which are expected to contain multiple-choice question-answer pairs. The `test_data` is truncated to 43 questions (`[40:83]`) for faster execution.
3.  **Prompt Building Functions:** Defines `build_prompt()` to construct prompts tailored to each of the `zero-shot`, `few-shot`, `knn-few-shot`, `cot`, and `self-consistency` strategies.
4.  **OpenAI API Wrapper:** Implements `call_openai()` with built-in retry logic to ensure robust communication with the OpenAI API.
5.  **K-NN Example Selection:** The `get_knn_examples()` function uses TF-IDF and cosine similarity to find the most relevant training examples for K-NN Few-shot prompting.
6.  **Prediction Parsing:** `extract_letter()` function extracts the predicted answer option (A, B, C, D, E) from the LLM's response.
7.  **Evaluation Loop:** The core of the notebook, this loop iterates through each defined model (`GPT-3.5`, `GPT-4.1`, `GPT-4o`) and each prompting technique. For each question in the test set, it:
    *   Constructs the appropriate prompt.
    *   Calls the OpenAI API to get a response.
    *   For `self-consistency`, it makes 5 API calls with temperature 0.7 and takes a majority vote.
    *   Records the prediction, actual answer, and correctness.
8.  **Results Saving:** After evaluating each model-technique pair, the accumulated results are saved to `medqa_eval_results.csv`.
9.  **Accuracy Summary:** A summary DataFrame showing the accuracy (in percentage) for each model-technique combination is printed to the console.

### Cell 2: Results Visualization

1.  **Load Pre-calculated Results:** This cell defines a `pandas.DataFrame` named `eval_results` with hardcoded accuracy values for different models and techniques. *Note: These values are pre-filled and not dynamically calculated from the CSV generated in Cell 1. They represent the expected outcomes from a full run.*
2.  **Plot Generation:** Uses `matplotlib` to create a bar chart visualizing the accuracy of each model across different prompting techniques.
3.  **Display DataFrame:** The `eval_results` DataFrame is printed below the plot.

## Key Insights and Results

Based on the hardcoded results in Cell 2:

*   **Few-shot Learning Dominates:** `Few-shot` and `K-NN Few-shot` techniques generally yield the highest accuracies across all models, demonstrating the significant benefit of providing in-context examples.
*   **GPT-4o's Performance:** `GPT-4o` exhibits superior performance in `few-shot` (72.09%) and `zero-shot` (55.81%) scenarios compared to `GPT-3.5` and `GPT-4.1` on this dataset subset.
*   **Unexpected CoT/Self-Consistency Performance:** Surprisingly, Chain-of-Thought (CoT) and Self-Consistency as implemented here show lower accuracies compared to simpler few-shot methods. This might indicate that for this specific task and prompt design, simpler direct examples are more effective, or the CoT reasoning output needs more sophisticated parsing.

```
Accuracy Summary (in %):
 technique      cot  few-shot  knn-few-shot  self-consistency  zero-shot
model
GPT-3.5        37.21     58.14         58.14             25.58      48.84
GPT-4.1        25.58     51.16         55.81             25.58      30.23
GPT-4o         25.58     72.09         65.12             25.58      55.81
```

## How to Use/Run the Notebook

1.  **Open in Google Colab:** Click the "Open in Colab" badge or directly upload the `.ipynb` file to Google Colab.
2.  **Upload Data:**
    *   Ensure you have your `train(1).jsonl` and `test(1).jsonl` dataset files.
    *   Upload these files to the Colab environment. The notebook expects them in the default `/content/` directory. You can drag and drop them into the file explorer pane in Colab.
3.  **Set OpenAI API Key:**
    *   Go to **"Tools"** -> **"Secrets"** in the Colab menu.
    *   Add a new secret variable named `OPEN_AI_KEY` and paste your OpenAI API key as its value. Ensure "Notebook access" is enabled for this secret.
4.  **Run Cells:**
    *   You can run each cell individually by clicking the play button next to it.
    *   Alternatively, go to **"Runtime"** -> **"Run all"** to execute the entire notebook.

The evaluation process will take some time depending on the number of test questions and API call latencies. Results will be saved to `medqa_eval_results.csv` and displayed in the notebook.