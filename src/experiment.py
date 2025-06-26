from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Tuple

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Enable 4-bit quantization
#     bnb_4bit_compute_dtype=torch.float16,  # Set compute dtype
#     bnb_4bit_use_double_quant=True,  # Enable double quantization for efficiency
# )

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path)#, quantization_config=quantization_config,)
device = torch.device('cuda')

model.to(device)

app = FastAPI()

# Input schema
class ChatRequest(BaseModel):
    messages: List[Tuple[str, str]]  # List of (role, content)
    max_new_tokens: int = 100

@app.post("/generate")
async def generate_text(query: ChatRequest):
    inputs = tokenizer.apply_chat_template([{'role': e[0], 'content': e[1] } for e in query.messages],
                                           add_generation_prompt=True, return_tensors='pt', return_dict=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=query.max_new_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": result}
