import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

model_dir = "/nfs/10.232.64.3/nvme3/kangyu/saved_models/final_answer_model/checkpoint-42"
tokenizer = AutoTokenizer.from_pretrained("/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE/saved_models/implicit-cot_cot-lora_llama2-13b_2e-4/checkpoint-168/merged/", trust_remote_code=True)
tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<unk>'})
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True).eval()

prompt = "Human: \n" + "{}" + "\n\nAssistant: \n"

def model_call(text, eos_token_ids):
    inputs = tokenizer([text], padding=True, return_tensors='pt').to("cuda")
    outputs = model.generate(**inputs, eos_token_id=eos_token_ids, do_sample=False)
    output = outputs[0]
    output = output[len(inputs['input_ids'][0]):]
    generated_texts = tokenizer.batch_decode([output], skip_special_tokens=True)

    return generated_texts[0]

with open("/nfs/a100-80G-17/kangyu/consistency_hallucinations/trytry/implicit_cot/data/gsm8k/train_cot-special-tokens_test.json", 'r') as f:
    data = json.load(f)

output = []
for one in tqdm(data):
    question = one['question']
    text = prompt.format(question)
    res = model_call(text, [tokenizer.eos_token_id])

    new_one = dict(one)
    new_one['response'] = res
    output.append(new_one)

dst_dir = "/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE/results"
with open(os.path.join(dst_dir, "lora-merged_final_answer_model-checkpoint-42.json"), 'w') as f:
    json.dump(output, f)

print("Done!")
