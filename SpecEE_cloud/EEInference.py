import argparse
import os
import re
from typing import Optional
import torch
import json
from EE_model import EEModel
from model_llama_ee import MLP

from fastchat.model import get_conversation_template
import time
from tqdm import trange
import json
from accuracy_prompt import get_commonsenseqa_prompt,get_mmlu_prompt,get_sst2_prompt
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer,AutoModelForCausalLM

def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    return dataset

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

def main(args):
    model = EEModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.draft_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
        predictor_path=args.predictor_path,
        pred_thresholds = args.pred_thresholds,
        # is_offload = False,
        # skip_model = "/home/xujiaming/xujiaming/research/ASPLOS-24/skip_layer/model.txt",
    )
    if args.task == 'speed':
        if args.dataset not in ['alpaca','gsm8k','mt_bench','sum','qa','humaneval']:
            print("Dataset "+args.dataset +" is not yet supported in "+args.task+" task!")
            exit(0)
        question_list = load_questions('./benchmark/'+args.dataset+'/question.jsonl',begin=0,end=80)
        exit_layer_id_list=[]
        output_ids_tot = 0
        st = time.time()
        torch.cuda.empty_cache()
        if args.enable_previous_cache:
            print("Enable previous cache!")
        for i in trange(len(question_list)):
            message = question_list[i]['turns'][0]
            conv = get_conversation_template("llama-2-chat")
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
            conv.append_message(conv.roles[0], message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "
            input_ids=model.tokenizer([prompt]).input_ids
            seqlen = len(input_ids[0])
            input_ids = torch.as_tensor(input_ids).cuda()
            output_ids=model(input_ids,max_new_tokens=256,exit_layer_id_list=exit_layer_id_list,enable_previous_cache=args.enable_previous_cache)
            output_ids_tot += len(output_ids[0]) - seqlen
            output=model.tokenizer.decode(output_ids[0])
        ed = time.time()
        spec = output_ids_tot/(ed-st)
        print('SpecEE '+ args.dataset + ' tokens per second :  ',spec)
        # print('average layer :  ',sum(exit_layer_id_list)/len(exit_layer_id_list))
        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        model = AutoModelForCausalLM.from_pretrained(args.base_model_path,torch_dtype=torch.float16,device_map="auto",attn_implementation="eager",low_cpu_mem_usage=True)
        model.eval()
        output_ids_tot = 0
        torch.cuda.empty_cache()
        st = time.time()
        for i in trange(len(question_list)):
            torch.cuda.empty_cache()
            message = question_list[i]['turns'][0]
            conv = get_conversation_template("llama-2-chat")
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
            conv.append_message(conv.roles[0], message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "
            input_ids=tokenizer([prompt]).input_ids
            seqlen = len(input_ids[0])
            input_ids = torch.as_tensor(input_ids).cuda()
            output_ids=model.generate(input_ids,max_new_tokens=256,do_sample=False)
            output_ids_tot += len(output_ids[0]) - seqlen
            output=tokenizer.decode(output_ids[0])
        ed = time.time()
        hf = output_ids_tot/(ed-st)
        print('HF '+args.dataset + ' tokens per second :  ',hf)
        print('SpecEE acceleration ratio is: ',spec/hf)
    elif args.task == 'accuracy':
        model.eval()
        if args.dataset not in ['commonsenseqa','sst2']:
            print("Dataset "+args.dataset +" is not yet supported in "+args.task+" task!")
            exit(0)
        if args.dataset == 'commonsenseqa':
            file_path = "./benchmark/commonsense_qa/data/validation-00000-of-00001.parquet"
            dataset = pq.read_table(file_path).to_pandas()
            correct = 0
            total = 0
            exit_layer_id_list=[]
            for _, row in dataset.iterrows():
                question = row['question']
                choices = row['choices']
                options = choices['label']
                answers = choices['text']
                correct_answer = row['answerKey'].strip()
                prompt = get_commonsenseqa_prompt(question,options,answers)
                input_ids=model.tokenizer([prompt]).input_ids
                input_ids = torch.as_tensor(input_ids).cuda()
                output_ids=model(input_ids,max_new_tokens=3,exit_layer_id_list=exit_layer_id_list,enable_previous_cache=args.enable_previous_cache)
                generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                answer_start_index = len(prompt+"Answer:")
                try:
                    predicted_answer = generated_text[answer_start_index:].strip()[0].upper()
                except:
                    predicted_answer = "N/A"
                if predicted_answer == correct_answer:
                    correct += 1
                if predicted_answer in ['A','B','C','D','E']:
                    total += 1
            accuracy = correct / total
            print(f"SpecEE Model's accuracy on comonsenseqa is: {accuracy:.2%}")
            torch.cuda.empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
            model = AutoModelForCausalLM.from_pretrained(args.base_model_path,torch_dtype=torch.float16,device_map="auto",attn_implementation="eager",low_cpu_mem_usage=True)
            model.eval()
            correct = 0
            total = 0
            for _, row in dataset.iterrows():
                question = row['question']
                choices = row['choices']
                options = choices['label']
                answers = choices['text']
                correct_answer = row['answerKey'].strip()
                prompt = get_commonsenseqa_prompt(question,options,answers)
                input_ids=tokenizer([prompt]).input_ids
                input_ids = torch.as_tensor(input_ids).cuda()
                output_ids=model.generate(input_ids,max_new_tokens=3,temperature=1e-6)
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                answer_start_index = len(prompt+"Answer:")
                try:
                    predicted_answer = generated_text[answer_start_index:].strip()[0].upper()
                except:
                    predicted_answer = "N/A"
                if predicted_answer == correct_answer:
                    correct += 1
                # if predicted_answer in ['A','B','C','D','E']:
                total += 1
            accuracy = correct / total
            print(f"HF Model's accuracy on comonsenseqa is: {accuracy:.2%}")
        elif args.dataset == 'sst2':
            file_path = "./benchmark/sst2/data/validation-00000-of-00001.parquet"  # 替换为您的数据集文件路径
            dataset = pq.read_table(file_path).to_pandas()
            exit_layer_id_list=[]
            total_time = 0
            output_ids_tot = 0
            total = 0
            correct = 0
            total = 0
            for _, row in dataset.iterrows():
                sentence = row['sentence']
                label = str(row['label']).strip()
                prompt = get_sst2_prompt(sentence)
                st = time.time()
                inputs = model.tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = torch.as_tensor(inputs).cuda()
                seqlen = len(inputs[0])
                outputs = model(input_ids, max_new_tokens=3,exit_layer_id_list=exit_layer_id_list)
                output_ids_tot += len(outputs[0]) - seqlen
                generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                ed = time.time()
                total_time += ed-st
                answer_start_index = len(prompt+":")
                try:
                    predicted_answer = generated_text[answer_start_index:].strip()[0]
                except:
                    predicted_answer = "N/A"
                if predicted_answer == label:
                    correct += 1
                total +=1
            print("SpecEE Model's accuracy on sst2 is: ",correct/total)
            torch.cuda.empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
            model = AutoModelForCausalLM.from_pretrained(args.base_model_path,torch_dtype=torch.float16,device_map="auto",attn_implementation="eager",low_cpu_mem_usage=True)
            model.eval()
            correct = 0
            total = 0
            for _, row in dataset.iterrows():
                sentence = row['sentence']
                label = str(row['label']).strip()
                prompt = get_sst2_prompt(sentence)
                st = time.time()
                inputs = tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = torch.as_tensor(inputs).cuda()
                seqlen = len(inputs[0])
                outputs = model.generate(input_ids, max_new_tokens=3,temperature=1e-6)
                output_ids_tot += len(outputs[0]) - seqlen
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                ed = time.time()
                total_time += ed-st
                answer_start_index = len(prompt+":")
                try:
                    predicted_answer = generated_text[answer_start_index:].strip()[0]
                except:
                    predicted_answer = "N/A"
                if predicted_answer == label:
                    correct += 1
                total += 1
            print("HF Model's accuracy on sst2 is: ",correct/total)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default="")
    parser.add_argument("--draft-model-path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="mt_bench")
    parser.add_argument("--task", type=str,choices=['speed', 'accuracy'], default="speed")
    parser.add_argument("--predictor-path", type=str, default="")
    parser.add_argument("--model-size", type=str,choices=['7B'],default="7B")
    parser.add_argument("--pred-thresholds", type=float,default=0.5)
    parser.add_argument("--enable-previous-cache", type=bool, default=False)


    args = parser.parse_args()
    main(args)
