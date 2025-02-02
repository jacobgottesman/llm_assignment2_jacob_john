import datasets
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import json 
import torch

# generate n_samples solutions per prompt
def generate_solutions(prompt, tokenizer, model, max_new_tokens=300, temperature=0.2, n_samples=20):

    # try to tokenize prompt to an mps tensor, if it fails, tokenize it to a CPU tensor
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

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

    return solutions

def test_problem(prompt, answer, tokenizer, model, max_tokens = 300, n_samples =20, temp=.2):
    # this generates solutions and returns the number of correct math solutions
    
    # generate solutions
    solutions = generate_solutions(prompt, tokenizer, model, max_new_tokens = max_tokens, n_samples = n_samples, temperature=temp)
    
    # get number of correct solutions
    num_correct = sum([1 if str(answer) == sol else 0 for sol in solutions])

    return num_correct

def test_problem_chain(prompt, answer, tokenizer, model, max_tokens = 300, n_samples =20, temp=.2):
    # this generates solutions and returns the number of correct math solutions for a chain of thought prompt

    # get solutions
    solutions = generate_solutions(prompt, tokenizer, model, max_new_tokens = max_tokens, n_samples = n_samples, temperature=temp)
    
    num_correct = 0
    for sol in solutions:
        
        # extact answer from response
        match = re.search(r'ANSWER[^0-9$+-]*\s*\$?([-+]?\d*\.?\d+)', sol)
        if match:
            extracted_answer = float(match.group(1))
            
            # check for correctness of answer
            if extracted_answer == answer:
                num_correct+=1

    return num_correct

def main():

    # load in data
    test_data = datasets.load_dataset("nuprl/engineering-llm-systems", "math_word_problems", split="test")

    MODEL = "/scratch/bchk/aguha/models/llama3p1_8b_base"
    DEVICE = "cuda"
    n_samp =5

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
    for problem in tqdm(test_data):

        # test zero shot
        num_correct += test_problem(f"Question: {problem['question']} \n Answer: ", problem['answer'], tokenizer, model, max_tokens=1, n_samples=n_samp, temp=.2)

    print(f'Zero Shot Accuracy: {round(num_correct/(len(test_data)*n_samp), 2)}')

    # few shot prompting
    num_correct = 0
    for problem in tqdm(test_data):

        # create multi shot prompt
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
        
        # test multi-shot
        num_correct += test_problem(few_shot_prompt, problem['answer'], tokenizer, model, n_samples = n_samp, max_tokens = 1)


    print(f'Few Shot Accuracy: {round(num_correct/(len(test_data)*n_samp), 2)}')

    num_correct = 0
    for problem in tqdm(test_data):
        
        # create chain of thought prompt
        chain_prompt = f"""
        Instruction: Solve the following problem using a step-by-step approach. Follow these steps:
        1. Identify the key information and given values.
        2. Break the problem into smaller subproblems if necessary.
        3. Perform the calculations systematically, explaining each step clearly.
        4. Double-check your calculations for correctness.
        5. At the end, output the final answer in the exact format:

           ANSWER: X

        Question: {problem['question']}

        Explanation:
        """

        # test chain of thought
        num_correct += test_problem_chain(chain_prompt, problem['answer'], tokenizer, model, n_samples = n_samp, max_tokens = 500)


    print(f'Chain of Thought Accuracy: {round(num_correct/(len(test_data)*n_samp), 2)}')

    
if __name__ == "__main__":
    main()


