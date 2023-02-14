from src.predict.generate_sentence import sentensizer_maker
import time


def cleaned_para(input_sentence,temp,generator):
  p=generator('<s>'+input_sentence+ '</s>>>>><p>',do_sample=True,max_length=len(input_sentence.split(" "))+200,temperature =temp,repetition_penalty=1.2)
  return p[0]['generated_text'].split('</s>>>>><p>')[1].split('</p>')[0]

def new_sampler(sentence,temp, model, tokenizer, device):
  #sentence = "I am a web developer with extensive knowledge of developing, enhancing, maintaining, and redesigning websites and mobile APPS."

  text = "<s>" + sentence + "</s>>>>><p>"

  encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")

  input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
  print(device)

  outputs = model.generate(input_ids=input_ids,
  attention_mask=attention_masks,
  max_length=512,
  temperature=temp,
  do_sample=True,
  top_k=256,
  top_p=0.95,
  early_stopping=True)

  for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

  return line.split(">>>><p>")[1].split("</p>")[0]



def paraphraser(text,temp,model, tokenizer, device):
  begin=time.time()
  x = [new_sampler(str(i), temp, model, tokenizer, device) for i in sentensizer_maker(text)]
  end=time.time()
  print(end-begin)
  return ("".join(x))
