from transformers import AutoModelForCausalLM, AutoTokenizer

AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")