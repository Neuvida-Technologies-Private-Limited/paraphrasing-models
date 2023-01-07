from transformers import pipeline


def model_creator(model,tokenizer,device):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return generator


