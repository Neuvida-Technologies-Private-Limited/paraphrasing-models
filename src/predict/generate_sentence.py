from nltk.tokenize import sent_tokenize



def sentensizer_maker(raw_text):
  sentences=sent_tokenize(raw_text)
  return sentences
