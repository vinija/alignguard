# ============================
# Module: scripts/train_deepspeed.py
# ============================
import deepspeed
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer



parser = argparse.ArgumentParser()
parser.add_argument('--deepspeed_config', type=str, required=True)
parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-2-7b')
args = parser.parse_args()



tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
model, _, _, _ = deepspeed.initialize(model=model, config=args.deepspeed_config)



print("Model initialized with DeepSpeed!")


