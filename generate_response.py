import torch
from transformers import GPT2Tokenizer
from train_gpt2 import GPT, GPTConfig

# Assuming 'GPT' is the class you defined and trained
# Load the model (replace 'path_to_trained_model' with your actual model path)
model = GPT(GPTConfig())
model.load_state_dict(torch.load(r'C:\Users\Admin\Documents\own_model.pth'))
model.eval()  # Set the model to evaluation mode
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load the tokenizer (we use GPT-2's tokenizer for this example)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def generate_text(model, tokenizer, prompt, max_length=50):
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate text (adjust max_length as needed)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated tokens to text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Example usage
prompt = "Good morning"
response = generate_text(model, tokenizer, prompt, max_length=100)
print(response)
