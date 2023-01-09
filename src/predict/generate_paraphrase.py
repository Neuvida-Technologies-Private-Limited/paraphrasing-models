from src.predict.generate_sentence import sentensizer_maker
import time


def cleaned_para(input_sentence,temp,generator):
  p=generator('<s>'+input_sentence+ '</s>>>>><p>',do_sample=True,max_length=len(input_sentence.split(" "))+200,temperature =temp,repetition_penalty=1.2)
  return p[0]['generated_text'].split('</s>>>>><p>')[1].split('</p>')[0]

def paraphraser(text,temp,generator):
  begin=time.time()
  x=[cleaned_para(str(i),temp,generator) for i in sentensizer_maker(text)]
  end=time.time()
  print(end-begin)
  return (".".join(x))
