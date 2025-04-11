import requests, json
from config import config

def trigger_fine_tuning_job():
    resource_name = config['azure']['openai_endpoint'].split("//")[1].split(".")[0]
    api_key = config['azure']['openai_api_key']
    training_file_url = config['azure']['fine_tune_file_url']
    base_model = "gpt-4"  # Base model to fine-tune
    hyperparameters = {
        "n_epochs": config['settings']['fine_tuning_n_epochs'],
        "batch_size": config['settings']['fine_tuning_batch_size'],
        "learning_rate_multiplier": config['settings']['fine_tuning_learning_rate_multiplier']
    }
    payload = {
        "training_file": training_file_url,
        "model": base_model,
        "hyperparameters": hyperparameters
    }
    api_version = "2021-06-01-preview"  # Update based on Azure docs
    endpoint_url = f"https://{resource_name}.openai.azure.com/openai/fine-tunes?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    response = requests.post(endpoint_url, headers=headers, json=payload)
    if response.status_code == 202:
        print("Fine-tuning job submitted successfully!")
        print("Job details:", json.dumps(response.json(), indent=4))
    else:
        print("Error submitting fine-tuning job:", response.status_code, response.text)

if __name__ == "__main__":
    trigger_fine_tuning_job()
