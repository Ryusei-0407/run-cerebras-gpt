# Program: VTSTech-GPT.py 2023-03-31 1:12:19 AM
# Description: Python script that generates text with Cerebras GPT pretrained and Corianas finetuned models 
# Author: Written by Veritas//VTSTech (veritas@vts-tech.org)
# GitHub: https://github.com/Veritas83
# Homepage: www.VTS-Tech.org
# Dependencies: transformers, colorama, Flask, torch
# pip install transformers colorama Flask torch
# Models are stored at C:\Users\%username%\.cache\huggingface\hub
import argparse
import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from colorama import Fore, Back, Style, init
from flask import Flask, request
global start_time, end_time, build, model_size, model_name, prompt_text
init(autoreset=True)
build="v0.2-r07"
tok=random.seed()
eos_token_id=tok
model_size = "111m"
model_name = "cerebras/Cerebras-GPT-111M"
parser = argparse.ArgumentParser(description='Generate text with Cerebras GPT models')
parser.add_argument('-m', '--model', choices=['111m', '256m', '590m','1.3b','2.7b','6.7b','13b'], help='Choose the model size to use (default: 111m)', type=str.lower)
parser.add_argument('-ce', '--cerb', action='store_true', help='Use Cerebras GPT pretrained models (default)')
parser.add_argument('-co', '--cori', action='store_true', help='Use Corianas finetuned models')
parser.add_argument('-cu', '--custom', type=str, help='Specify a custom model')
parser.add_argument('-p', '--prompt', type=str, default="AI is", help='Text prompt to generate from (default: "AI is")')
parser.add_argument('-s', '--size', type=int, default=256)
parser.add_argument('-l', '--length', type=int, default=256)
parser.add_argument('-tk', '--topk', type=float, default=None)
parser.add_argument('-tp', '--topp', type=float, default=None)
parser.add_argument('-ty', '--typp', type=float, default=None)
parser.add_argument('-tm', '--temp', type=float, default=None)
parser.add_argument('-t', '--time', action='store_true', help='Print execution time')
parser.add_argument('-c', '--cmdline', action='store_true', help='cmdline mode, no webserver')
args = parser.parse_args()
if args.model:
	model_size = args.model
if args.prompt:
	prompt_text = args.prompt
top_p = args.topp
top_k = args.topk
typ_p = args.typp
temp = args.temp

def get_model():
	global model_size, model_name
	if args.cori:
		if model_size == '111m':
			model_name = "Corianas/111m"
		elif model_size == '256m':
			model_name = "Corianas/256m"
		elif model_size == '590m':
			model_name = "Corianas/590m"	    
		elif model_size == '1.3b':
			model_name = "Corianas/1.3B"	    
		elif model_size == '2.7b':
			model_name = "Corianas/2.7B"	    
		elif model_size == '6.7b':
			model_name = "Corianas/6.7B"	    
		elif model_size == '13b':
			model_name = "Corianas/13B"	    
	elif args.cerb or not args.cmdline:
		if model_size == '111m':
			model_name = "cerebras/Cerebras-GPT-111M"
		elif model_size == '256m':
			model_name = "cerebras/Cerebras-GPT-256M"
		elif model_size == '590m':
			model_name = "cerebras/Cerebras-GPT-590M"
		elif model_size == '1.3b':
			model_name = "cerebras/Cerebras-GPT-1.3B"
		elif model_size == '2.7b':
			model_name = "cerebras/Cerebras-GPT-2.7B"
		elif model_size == '6.7b':
			model_name = "cerebras/Cerebras-GPT-6.7B"
		elif model_size == '13b':
			model_name = "cerebras/Cerebras-GPT-13B"
	elif args.custom:
		  model_name = args.custom	    	
	return model_name
	
model_name = get_model()
max_length = int(args.length)

def banner():
    global model_name
    print(Style.BRIGHT + f"VTSTech-GPT {build} - www: VTS-Tech.org git: Veritas83")
    print("Using Model : " + Fore.RED + f"{model_name}")
    print("Using Prompt: " + Fore.YELLOW + f"{prompt_text}")
    print("Using Params: " + Fore.YELLOW + f"max_new_tokens:{max_length} do_sample:True use_cache:True no_repeat_ngram_size:2 top_k:{top_k} top_p:{top_p} typical_p:{typ_p} temp:{temp}")
def CerbGPT(prompt_text):
    global start_time, end_time, build, model_size, model_name	
    temp=None
    top_k=None
    top_p=None
    start_time = time.time()	
    #model_name = get_model()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    opts = {}
    if temp is not None:
        opts["temperature"] = temp
    if top_k is not None:
        opts["top_k"] = top_k
    if top_p is not None:
        opts["top_p"] = top_p        
    if typ_p is not None:
        opts["typical_p"] = typ_p        
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generated_text = pipe(prompt_text, max_new_tokens=max_length, do_sample=True, use_cache=True, no_repeat_ngram_size=2, **opts)[0]
    
    end_time = time.time()
    return generated_text['generated_text']


if not args.cmdline:
	app = Flask(__name__)

	@app.route('/', methods=['GET'])
	def index():
	    return  f"<p>VTSTech-GPT {build} - <a href=https://www.VTS-Tech.org>www.VTS-Tech.org</a> <a href=https://github.com/Veritas83>github.com/Veritas83</a><br><br>ie: <a href=http://localhost:5000/generate?model=256m&prompt=AI%20is>Prompt: AI is</a>"	    
	@app.route('/generate', methods=['GET'])
	def generate():
	    global model_size
	    model_size = request.args.get('model', '111m')
	    model_name = get_model()
	    prompt_text = request.args.get('prompt', 'AI is')
	    generated_text = CerbGPT(prompt_text)
	    generated_text = f"<p>VTSTech-GPT {build} - <a href=https://www.VTS-Tech.org>www.VTS-Tech.org</a> <a href=https://github.com/Veritas83>github.com/Veritas83</a><br><br>Using Model : <b><a href=https://huggingface.co/{model_name}>{model_name}</a></b><br>Using Prompt: <i>{prompt_text}</i><br>Using Params: max_length:{max_length} top_k:{top_k} top_p:{top_p} temp:{temp}<br><br>" + generated_text + f"<br><br>Execution time: {end_time - start_time:.2f} seconds</p>"	    
	    return generated_text

if __name__ == '__main__':
    global start_time, end_time	    
    if args.cmdline:
    	banner()
    	print(CerbGPT(prompt_text))
    	if args.time:
    		print(Style.BRIGHT + Fore.RED + f"Script finished. Execution time: {end_time - start_time:.2f} seconds")
    else:
    	app.run(host='0.0.0.0', port=5000)