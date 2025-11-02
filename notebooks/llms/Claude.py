import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Load model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

# Example email text
email_text = """
From: john.smith@company.com
To: sarah.jones@company.com
Subject: Project Update - Q4 Deliverables

Hi Sarah,

I wanted to update you on the Q4 project timeline. The deadline is December 15, 2024, 
and we need to deliver the API integration module. The budget allocated is $50,000.

Please confirm if you can lead the frontend team for this initiative.

Best regards,
John Smith
Senior Developer
Phone: +1-555-0123
"""

# Define the JSON schema we want to extract
schema = {
    "type": "object",
    "properties": {
        "sender_email": {"type": "string"},
        "recipient_email": {"type": "string"},
        "subject": {"type": "string"},
        "deadline": {"type": "string"},
        "budget": {"type": "string"},
        "sender_name": {"type": "string"},
        "sender_role": {"type": "string"},
        "sender_phone": {"type": "string"}
    }
}

# Format the prompt using Llama 3.2 chat template
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that extracts information from emails and returns valid JSON. Only return the JSON object, no additional text or explanation."
    },
    {
        "role": "user",
        "content": f"""Extract the following information from the email below and return it as a JSON object matching this schema:

{json.dumps(schema, indent=2)}

Email:
{email_text}

Return only the JSON object:"""
    }
]

# Apply chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,  # Low temperature for more deterministic output
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("Model Response:")
print(response)

# Try to parse as JSON
try:
    # Clean up response if needed
    json_start = response.find('{')
    json_end = response.rfind('}') + 1
    if json_start != -1 and json_end > json_start:
        json_str = response[json_start:json_end]
        parsed_json = json.loads(json_str)
        print("\n\nParsed JSON:")
        print(json.dumps(parsed_json, indent=2))
    else:
        print("\n\nCould not find valid JSON in response")
except json.JSONDecodeError as e:
    print(f"\n\nJSON parsing error: {e}")