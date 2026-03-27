from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./merged-model-directory"

# Load your custom LRP-merged model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Test generation
inputs = tokenizer("The secret to time travel is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
