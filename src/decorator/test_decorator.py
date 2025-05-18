import json
from openai import OpenAI
from src.decorator.decorator import mask_pii

client = OpenAI(api_key="")

# --- Simulate a message from a client ---
def receive_client_message():
    return (
        "Hi, I'm Andrea Rossi. I had a car accident on April 2nd, 2024 in Milan. "
        "You can reach me at +393473761898. I'd like legal assistance regarding damages."
    )

# --- Automatically mask PII and summarize using OpenAI ---
@mask_pii()  # assumes model loading inside
def process_intake(masked_message):
    
    # Use OpenAI to generate a summary for internal case tracking
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": ("You are an assistant helping a legal office structure client messages."
                                 " Please return a summary of the legal task required and the client's name in JSON format: "
                                 "{"
                                 "  \"task\": \"string\","
                                 "  \"client_name\": \"string\""
                                 "}")
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Message: {masked_message}"
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "text"
            }
        },
        temperature=0.1
    )
    summary = json.loads(response.output[0].content[0].text.replace("```json", "").replace("```", ""))
    print(">> GPT-4 Summary:", summary)
    return summary

# --- Run the pipeline ---
if __name__ == "__main__":
    raw_message = receive_client_message()
    process_intake(raw_message)
