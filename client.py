import time
from openai import OpenAI


CLIENT = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)
N = 100

start = time.time()
for i in range(N):
    completion = CLIENT.chat.completions.create(
    model="runs/quantized",
    messages = [
            {
                "role": "system",
                "content": "You are an ai image generator that takes user requests and interprets and converts them to optimized stable diffusion prompts. Always specify if the image is a photograph/painting/digital art/etc.",
            },
            {
                "role": "user",
                "content": "a photo of donald trump doing a handstand, 32k, HD, best quality",
            },
        ]
    )
    print(completion.choices[0].message.content)
print((time.time() - start) / N)
