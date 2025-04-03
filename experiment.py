from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_path = "NovaSky-AI/Sky-T1-7B-Zero"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Set compute dtype
    bnb_4bit_use_double_quant=True,  # Enable double quantization for efficiency
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config,)
device = torch.device('cuda')

model.to(device)


input_texts = ['I want you to make a descision about what to do in this situation: I have 3 docker images and a load of 80\% cpu usage on server 1 and 1 docker image and a load of 20% cpu usage on server 2. do you 1: move 1 image, 2: move 2 images, 3: do nothing', 
               'please generate an example of a JSON structure',
               'I want to ask you questions and get responses in a json format with the some integers detailing the answer to my question and some reasoning to why those where chosen. how can i best ask you to do so?',
               '''{
  "question": "What is the capital of France?",
  "options": [
    "Paris",
    "London",
    "Berlin",
    "Rome"
  ],
  "correct_answer": "?",
  "reasoning": "?"
}
''',
'''
can you copy this format and fill in the question marks for me? 
\\boxed{
  "question": "What is the capital of France?",
  "options": [
    "Paris",
    "London",
    "Berlin",
    "Rome"
  ],
  "correct_answer": "?",
  "reasoning": "?"
}
'''
]

'''En dan nog dit'''

print(tokenizer.chat_template)

# for input_text in input_texts:
#     inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': input_text }],
#                                            add_generation_prompt=True, return_tensors='pt', return_dict=True).to(device)

#     outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.6)

#     print(tokenizer.decode(outputs[0], skip_special_tokens=True))
#     print("============================ END ===============================")

