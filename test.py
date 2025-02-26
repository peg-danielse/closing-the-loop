# Use a pipeline as a high-level helper
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Set compute dtype
    bnb_4bit_use_double_quant=True,  # Enable double quantization for efficiency
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config,)
device = torch.device('cuda')

model.to(device)


input_texts = ['I want you to make a descision about what to do in this situation: I have 3 docker images and a load of 80\% cpu usage on server 1 and 1 docker image and a load of 20% cpu usage on server 2. do you 1: move images, 2: do nothing',
               'I want you to think of a format to make a choice in conserning moving docker image from one server to another in order to distribute load'] #["When I grow up I will be a clouds engineer", 'what model are you?', 'can you think of a nickname for my wife: judy?']


for input_text in input_texts:
    inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': "can you respond in this json format: { \'choice\': 0, \'reason\': \'\' }" },
                                            {'role': 'user', 'content': input_text}], add_generation_prompt=True, return_tensors='pt', return_dict=True).to(device)

    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], pad_token_id=tokenizer.eos_token_id, max_length=500, temperature=0.7)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("============= END ==============")

