from transformers import pipeline


def model_creator(model,tokenizer):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    return generator


