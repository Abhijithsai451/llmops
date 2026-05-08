import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


model_id = os.getenv("MODEL_ID")
saved_model = os.getenv("SAVED_MODEL")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

state_dict = torch.load(saved_model, map_location="cpu")
model.load_state_dict(state_dict)

model.to(device)
model.eval()

def generate(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, max_length = 50)
    return tokenizer.decode(outputs[0])

model.save_pretrained("./gpt2-finetuned")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("./gpt2-finetuned")




