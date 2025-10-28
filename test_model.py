import json
from llm_generator import LLMGenerator

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

openai_config = config["providers"]["openai"]

print("Configuration:")
print(f"  API Key: {openai_config['api_key'][:20]}...")
print(f"  Model: {openai_config['model_id']}")
print(f"  Base URL: {openai_config['base_url']}")
print()

generator = LLMGenerator(openai_config)

print("="*50)
print("Testing OpenAI API:")

test_prompt = "Who are you? What model are you? What is your specific model name?"
system_message = "You are a helpful AI assistant."

try:
    response = generator.generate(test_prompt, system_message)
    if response:
        print("✅ API call successful!")
        print(f"Response: {response}")
    else:
        print("❌ API call failed: Empty response")
except Exception as e:
    print(f"❌ API call failed: {e}")

print("="*50)