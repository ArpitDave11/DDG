import openai, logging
from config import config

def generate_description(table: str, column: str, datatype: str, examples: list[dict]) -> str:
    """Generate a description for the given attribute using GPT-4, conditioned on similar examples."""
    deployment = config['azure']['openai_deployment_gpt4']  # GPT-4 deployment name
    # System instruction to guide the model's style and behavior
    system_prompt = ("You are a knowledgeable data dictionary assistant. "
                     "Write clear, concise descriptions for database columns.")
    # Construct messages for OpenAI ChatCompletion
    messages = [{"role": "system", "content": system_prompt}]
    # Include each example as a user query and assistant answer
    for ex in examples:
        # Ensure example prompt and completion are present
        prompt_text = ex.get('prompt') or ""
        completion_text = ex.get('completion') or ex.get('description') or ""
        if prompt_text and completion_text:
            # Treat the example prompt as a user message and the description as assistant message
            messages.append({"role": "user", "content": prompt_text})
            messages.append({"role": "assistant", "content": completion_text})
    # Now add the new attribute prompt as the final user query
    new_prompt = f"Column: {table}.{column} (Type: {datatype})"
    messages.append({"role": "user", "content": new_prompt})
    logging.info(f"Generating description for {table}.{column} ...")
    try:
        response = openai.ChatCompletion.create(
            engine=deployment,
            messages=messages,
            temperature=config['settings']['temperature'],
            max_tokens=config['settings']['max_tokens']
        )
    except Exception as e:
        logging.error(f"OpenAI API error during description generation: {e}")
        raise
    # Extract the assistant's answer
    content = response['choices'][0]['message']['content']
    description = content.strip()
    return description
