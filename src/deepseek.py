import os
import openai

# Configure the OpenAI client to use the Azure OpenAI service
openai.api_type = "azure"
openai.api_base = "https://azureopenai-regulosity-llm.openai.azure.com/"
openai.api_version = "2024-12-01-preview"
openai.api_key = "e9301c7d4dcd4575a89778297fbed7f2"

# Create a chat completion request using the deployed GPT-4.1 model.
# Note: In Azure OpenAI, use the "engine" parameter with the deployment name.
response = openai.ChatCompletion.create(
    engine="gpt-4.1",  # Replace with the actual deployment name if different
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is the one most unique thing about human experience?"}
    ],
    temperature=0.1,
    top_p=0.1
)

print(response.choices[0].message["content"])
