import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

model_dir = "/nfs/a100-80G-15/kangyu/saved_models/Llama-2-13b-257-cot-tokens/checkpoint-168/"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<unk>'})
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True).eval()

belle_prompt = "Human: \n" + "{}" + "\n\nAssistant: \n"

START_COT_ID_0 = 32000
END_COT_ID = 32256
COT_LEN = 50
instruction = "[INST] " + "{}" + " [/INST]"

def model_call(eos_token_ids, inputs=None, text=None):

    if inputs is None:
        inputs = tokenizer([text], padding=True, return_tensors='pt').to("cuda")
    else:
        inputs = {"input_ids": inputs}

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
    text = instruction.format(question)
    instruct_id = tokenizer.encode(text, add_special_tokens=False)
    input_cot_id = []
    for i in range(COT_LEN):
        input_cot_id.append(START_COT_ID_0 + i)
    input_cot_id = input_cot_id + [END_COT_ID]

    input_id = instruct_id + input_cot_id
    input_id = [tokenizer.bos_token_id] + input_id
    input_ids = torch.tensor([input_id]).to("cuda")

    res = model_call(eos_token_ids=[tokenizer.eos_token_id], inputs=input_ids)

    new_one = dict(one)
    new_one['response'] = res
    output.append(new_one)

dst_dir = "/nfs/a100-80G-17/kangyu/consistency_hallucinations/BELLE/results"
with open(os.path.join(dst_dir, "257-cot-tokens_first-round_COT_LEN-{}.json".format(COT_LEN)), 'w') as f:
    json.dump(output, f)

print("Done!")


