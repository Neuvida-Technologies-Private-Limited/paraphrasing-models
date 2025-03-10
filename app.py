from flask import Flask,request,jsonify
from src.predict.generate_paraphrase import paraphraser
from transformers import GPT2Tokenizer,OPTForCausalLM
#from src.predict.generate_model import model_creator
import os
import torch

path=os.getcwd()

device=torch.device(0)
print(device)

model = OPTForCausalLM.from_pretrained(path+"/models/opt_paraphraser").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b",truncation=True,padding=True)

#generator = model_creator(model, tokenizer,device)

print("hello")

app = Flask(__name__)  # initialising the flask


@app.route('/para', methods=["GET","POST"]) # route with allowed methods as POST and GET
def index():
    if request.method == 'POST':
        print('hello')
        torch.cuda.empty_cache()
        input_txt=request.json
        print(input_txt)
        output_txt=paraphraser(input_txt['content'],float(input_txt['temperature']),model, tokenizer, device)
        print(output_txt)
        torch.cuda.empty_cache()
        return jsonify({"output":output_txt})  # showing the review to the user


@app.route('/train', methods=['GET','POST'])
def train():
    if request.method == 'POST':
        print("under_construction")



if __name__ == "__main__":
  port = int(os.environ.get('PORT', 8080))
  app.run(debug=True, host='0.0.0.0', port=port)
    