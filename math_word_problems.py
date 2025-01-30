import datasets
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import json 
import torch

# generate 20 solutions per prompt
def generate_solutions(prompt, tokenizer, model, max_new_tokens=300, temperature=0.2, n_samples=20):

    # try to tokenize prompt to an mps tensor, if it fails, tokenize it to a CPU tensor
    try:
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("mps")
    except:
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]

    # generate n_samples number of solutions
    outputs = model.generate(
        inputs,
        pad_token_id = tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=None,
        num_return_sequences=n_samples)
    
    # only decode the new tokens after the input prompt
    outputs = [output[inputs.shape[1]:] for output in outputs]
    solutions = [tokenizer.decode(output) for output in outputs]

    # clip solution text after the end of the function
    pattern = "|".join(["\ndef", "\nclass", "\nif", "\nprint", "\n"])
    clipped_solutions = [re.split(pattern, solution)[0] for solution in solutions]

    return clipped_solutions

def test_problem(prompt, answer, tokenizer, model):
    # this generates solutions and returns the number of correct math solutions

    solutions = generate_solutions(prompt, tokenizer, model)

    num_correct = sum([1 if sol == answer else 0 for sol in solutions])

    return num_correct

def main():

    # load in data
    test_data = datasets.load_dataset("nuprl/engineering-llm-systems", "math_word_problems", split="test")

    MODEL = "/scratch/bchk/aguha/models/llama3p1_8b_base"
    DEVICE = "cuda"

    # load in tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token 

    # load in model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
    ).to(device=DEVICE)

    # Zero shot prompting
    num_correct = 0
    for problem in test_data:

        num_corect += test_problem(problem['question'], problem['answer'], tokenizer, model)

    print(f'Zero Shot Accuracy: {round(num_correct/(len(test_data)*20), 2)}')

    # few shot prompting
    num_correct = 0
    for problem in test_data:

        few_shot_prompt = f"""Instruction: Solve the following math word problem and provide only the final numerical answer. Do not include explanations or steps.

                    Examples:
                    Q1: A train travels 120 miles in 3 hours. What is its average speed in miles per hour?
                    A1: 40

                    Q2: If a rectangle has a length of 8 cm and a width of 5 cm, what is its area in square centimeters?
                    A2: 40

                    Q3: A store sells apples for $1.25 each. If you buy 4 apples, how much do you pay in total?
                    A3: 5.00
                    
                    Q4: {problem['question']}
                    A4: """

        num_corect += test_problem(few_shot_prompt, problem['answer'], tokenizer, model)

    print(f'Few Shot Accuracy: {round(num_correct/(len(test_data)*20), 2)}')

    # Zero shot prompting
    num_correct = 0
    for problem in test_data:
        chain_prompt = f"""Instruction: Solve the following math word problem step by step, reasoning through the solution before providing the final numerical answer.
                         Question: {problem['question']},
                         Answer: """

        num_corect += test_problem(chain_prompt, problem['answer'], tokenizer, model)

    print(f'Chain-of-though Accuracy: {round(num_correct/(len(test_data)*20), 2)}')


