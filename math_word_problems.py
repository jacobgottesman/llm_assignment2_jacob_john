import sys
import datasets
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import namedtuple
from pprint import pprint
from tqdm.auto import tqdm

ZeroShotOut = namedtuple("ZeroShotOut", ["question", "answer", "completion", "solution", "score"])
FewShotOut = namedtuple("FewShotOut", ["question", "answer", "completion", "solution", "score"])
COTOut = namedtuple("COTOut", ["question", "answer", "completion", "solution", "score"])
PALOut = namedtuple("PALOut", ["question", "answer", "completion", "solution", "score", "error"])

MODEL = "/scratch/bchk/aguha/models/llama3p1_8b_base"
DEVICE = "cuda"

ZERO_SHOT_PROMPT = f"""Instruction: Solve the following math word problem and provide only the final numerical answer. Do not include explanations or steps."""
FEW_SHOT_PROMPT = f"""Instruction: Solve the following math word problem and provide only the final numerical answer. Do not include explanations or steps.

Examples:
Question: A train travels 120 miles in 3 hours. What is its average speed in miles per hour?

Answer: 40


Question: If a rectangle has a length of 8 cm and a width of 5 cm, what is its area in square centimeters?

Answer: 40


Question: A store sells apples for $1.25 each. If you buy 4 apples, how much do you pay in total?

Answer: 5.00
"""
COT_PROMPT = f"""Instruction: Solve the following problem using a step-by-step approach. Follow these steps:
    1. Identify the key information and given values.
    2. Break the problem into smaller subproblems if necessary.
    3. Perform the calculations systematically, explaining each step clearly.
    4. Double-check your calculations for correctness.
    5. At the end, output the final answer in the exact format:
        Answer: X

Examples:
Question: A train travels 120 miles in 3 hours. What is its average speed in miles per hour?

Reasoning: The train traveled 120 miles in 3 hours. To calculate average speed, we divide the distance by the time. Therefore, our answer is 120 / 3

Answer: 40


Question: A rectangle has a length of 8 cm and a width of 5 cm. What is the sum of its perimeter and area?

Reasoning: The perimeter of a rectangle is p = 2 * (length + width). Therefore, the perimeter of the rectangle is p = 2 * (8 + 5) = 26. The area of a rectangle is a = l * w. Therefore, the area of the rectangle is a = 8 * 5 = 40. Therefore, our answer is 26 + 40 = 66.

Answer: 66


Question: Larry is buying water bottles for the next two weeks. For the first week, Larry drinks 2 bottles of water a day. For the second week, Larry drinks 2 times the amount of water he drank the first week. If a crate of water bottles has 21 water bottles, how many crates of water bottles does Larry have to buy for two weeks?

Reasoning: In week 1, Larry drinks 2 bottles of water a day. Therefore, in week 1, Larry drinks 2 * 7 = 14 bottles of water. In week 2, Larry drinks 2 times the amount of water he drank in week 1. Therefore, in week 2, Larry drinks 14 * 2 = 28 bottles of water. In total, Larry drinks 14 + 28 = 42 bottles of water. If each crate has 21 water bottles, then Larry must buy 42 / 21 = 2 crates of water bottles in total.

Answer: 2


Question: A recipe calls for 2 cups of flour to make 12 cookies. How many cups of flour are needed to make 30 cookies?

Reasoning: 2 cups of flour are required to make 12 cookies. Therefore, 2 / 12 = 1/6 cups of flour are required to make 1 cookie. Therefore, to make 30 cookies, we need 30 * 1/6 = 5 cups of flour in total.

Answer: 5


Question: It takes 3 workers 6 hours to paint a house. If 9 workers paint at the same rate, how long will it take them to paint the same house?

Reasoning: It took 3 workers 6 hours to paint a house. Therefore, a total of 3 * 6 = 18 total worker hours were required to paint the house. If we had 9 workers who were working at the same pace, then it would take 18 / 9 = 2 hours to paint the house.

Answer: 2


Question: A password consists of 3 letters followed by 2 digits. If letters can be A-Z and digits can be 0-9, how many different passwords can be formed?

Reasoning: Each of the first three characters in the password is a letter A-Z. Therefore, for each of the first three characters there are 26 options. Each of the last two characters in the password is a digit 0-9. Therefore, for each of the last two characters there are 10 options. Therefore, all together, there are 26 * 26 * 26 * 10 * 10 = 1757600 different possible passwords.

Answer: 1757600


Question: A store is having a 25% off sale. If a shirt originally costs $80, and there is also a 8% sales tax, how much will the shirt cost in total?

Reasoning: The original price of the shirt was $80. If the store is having a 25% off sale, then the sale price of the shirt is $80 * 0.75 = $60. However, there is also an 8% sales tax. Therefore, the final price of the shirt is $60 * 1.08 = $64.8

Answer: 64.8


Question: Marie ordered one chicken meal that costs $12, 5 packs of milk that cost $3 each, 4 apples that cost $1.50 each, and some boxes of pizza. Marie paid a total of $50. How many boxes of pizza did Marie order if each box costs $8.50?

Reasoning: Marie bought one chicke mean for $12. Therefore, Marie spent $12 on chicken. Marie bought 5 packs of milk for $3 each. Therefore, marie spent $15 on milk. Marie bought 4 applies for $1.50 each. Therefore, Marie spent $6 on apples. Therefore, Marie spent $12 + $15 + $6 = $33 on chicken, milk, and apples. If Marie spent $50 in total, then she spent $50 - $33 = $17 on boxes of pizza. If each box of pizza costs $8.50, then Marie bought $17 / $8.50 = 2 boxes of pizza.

Answer: 2
"""
PAL_PROMPT = f"""Instruction: Solve each of the following math word problems by filling in function solve() with python code that solves the problem. Be sure to approach the problem step by step.

Examples:
Question: A train travels 120 miles in 3 hours. What is its average speed in miles per hour?
Let's solve this step by step!
Answer:
def solve():
    total_miles = 120
    total_hours = 3
    avg_speed = total_miles / total_hours
    return avg_speed

Question: A rectangle has a length of 8 cm and a width of 5 cm. What is the sum of its perimeter and area?
Let's solve this step by step!
Answer:
def solve():
    length = 8
    width = 5
    perimeter = 2 * (length + width)
    area = length * width
    return perimeter + area

Question: Larry is buying water bottles for the next two weeks. For the first week, Larry drinks 2 bottles of water a day. For the second week, Larry drinks 2 times the amount of water he drank the first week. If a crate of water bottles has 21 water bottles, how many crates of water bottles does Larry have to buy for two weeks?
Let's solve this step by step!
Answer:
def solve():
    bottles_per_day_week_one = 2
    total_bottles_week_one = bottles_per_day_week_one * 7
    total_bottles_week_two = total_bottles_week_one * 2
    total_bottles = total_bottles_week_one + total_bottles_week_two
    bottles_per_crate = 21
    crates_needed = total_bottles / bottles_per_crate
    return crates_needed

Question: A recipe calls for 2 cups of flour to make 12 cookies. How many cups of flour are needed to make 30 cookies?
Let's solve this step by step!
Answer:
def solve():
    flour_per_cookie = 2 / 12
    total_flour = flour_per_cookie * 30
    return total_flour

Question: It takes 3 workers 6 hours to paint a house. If 9 workers paint at the same rate, how long will it take them to paint the same house?
Let's solve this step by step!
Answer:
def solve():
    total_worker_hours = 3 * 6
    num_workers = 9
    time_needed = total_worker_hours / num_workers
    return time_needed

Question: A password consists of 3 letters followed by 2 digits. If letters can be A-Z and digits can be 0-9, how many different passwords can be formed?
Let's solve this step by step!
Answer:
def solve():
    num_letters = 26 ** 3
    num_digits = 10 ** 2
    total_passwords = num_letters * num_digits
    return total_passwords

Question: A store is having a 25% off sale. If a shirt originally costs $80, and there is also a 8% sales tax, how much will the shirt cost in total?
Let's solve this step by step!
Answer:
def solve():
    original_price = 80
    discount_amount = 80 * 0.25
    discount_price = original_price - discount_amount
    tax_amount = discount_price * 0.08
    final_price = discount_price + tax_amount
    return final_price

Question: Marie ordered one chicken meal that costs $12, 5 packs of milk that cost $3 each, 4 apples that cost $1.50 each, and some boxes of pizza. Marie paid a total of $50. How many boxes of pizza did Marie order if each box costs $8.50?
Let's solve this step by step!
Answer:
def solve():
    chicken_meal_cost = 12
    milk_cost = 5 * 3
    apples_cost = 4 * 1.50
    subtotal = chicken_meal_cost + milk_cost + apples_cost
    remaining = 50 - subtotal
    pizza_boxes = remaining / 8.50
    return pizza_boxes
"""

# split list into sublists of size x
def split_list(l, x):
    return [l[i:i + x] for i in range(0, len(l), x)]

# extract number from string and return as integer
def extract_number(completion):
    match = re.search(r'\s(\d+(\.\d+)?)', completion)
    return round(float(match.group(1))) if match else completion

def generate_batch(prompts, tokenizer, model, kwargs):
    # pad inputs to accomodate batch, load inputs to gpu
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs)
    # decode tokens after input prompt tokens
    outputs = [output[inputs["input_ids"].shape[1]:] for output in outputs]
    return [tokenizer.decode(output) for output in outputs]

def zero_shot(model, tokenizer, batch):
    # build prompt batch
    questions = [problem["question"] for problem in batch]
    prompts = [f"{ZERO_SHOT_PROMPT}\n\nQuestion: {question}\n\nAnswer:" for question in questions]

    # generate completions
    completions = generate_batch(prompts, tokenizer, model, { "max_new_tokens": 5, "do_sample": True, "temperature": 0.2, "top_p": None, "num_return_sequences": 1 })
    zero_shot_out = []
    for problem, completion in zip(batch, completions):
        # extract integer answer
        solution = extract_number(completion)
        if solution != problem["answer"]:
            # incorrect solution
            out = ZeroShotOut(problem["question"], problem["answer"], completion, solution, 0)
            zero_shot_out.append(out)
        else:
            # correct solution
            out = ZeroShotOut(problem["question"], problem["answer"], completion, solution, 1)
            zero_shot_out.append(out)
    return zero_shot_out

def few_shot(model, tokenizer, batch):
    # build prompt batch
    questions = [problem["question"] for problem in batch]
    prompts = [f"{FEW_SHOT_PROMPT}\n\nQuestion: {question}\n\nAnswer:" for question in questions]

    # generate completions
    completions = generate_batch(prompts, tokenizer, model, { "max_new_tokens": 5, "do_sample": True, "temperature": 0.2, "top_p": None, "num_return_sequences": 1 })
    few_shot_out = []
    for problem, completion in zip(batch, completions):
        # extract integer answer
        solution = extract_number(completion)
        if solution != problem["answer"]:
            # incorrect solution
            out = FewShotOut(problem["question"], problem["answer"], completion, solution, 0)
            few_shot_out.append(out)
        else:
            # correct solution
            out = FewShotOut(problem["question"], problem["answer"], completion, solution, 1)
            few_shot_out.append(out)
    return few_shot_out

def cot_process_completion(completion):
    # match string Answer: x where x is some number and return x as an integer
    match = re.search(r'Answer[^0-9$+-]*\s*\$?([-+]?\d*\.?\d+)', completion)
    if match:
        return round(float(match.group(1)))
    return completion.strip()

def cot(model, tokenizer, batch):
    # build prompt batch
    questions = [problem["question"] for problem in batch]
    prompts = [f"{COT_PROMPT}\n\nQuestion: {question}\n\nReasoning:" for question in questions]

    # generate completions
    completions = generate_batch(prompts, tokenizer, model, { "max_new_tokens": 300, "do_sample": True, "temperature": 0.2, "top_p": 0.9, "top_k": 50, "num_return_sequences": 1 })
    cot_out = []
    for problem, completion in zip(batch, completions):
        # extract integer answer
        solution = cot_process_completion(completion)
        if solution != problem["answer"]:
            # incorrect solution
            out = COTOut(problem["question"], problem["answer"], completion, solution, 0)
            cot_out.append(out)
        else:
            # correct solution
            out = COTOut(problem["question"], problem["answer"], completion, solution, 1)
            cot_out.append(out)
    return cot_out

def pal_execute_completion(completion):
    # load function string into python object, execute, and return solution
    try:
        exec(completion)
        solution = round(eval("solve()"))
        return solution, None
    except Exception as e:
        return None, e

def pal_process_completion(tokenizer, completion):
    # extract the function string from model completion (before 'Question' text or 'end of string' text)
    split_index = min(
        completion.find("Question") if "Question" in completion else float("inf"),
        completion.find(tokenizer.eos_token) if tokenizer.eos_token in completion else float("inf")
    )
    return completion[:split_index].strip() if split_index != float("inf") else completion.strip()

def pal(model, tokenizer, batch):
    # build prompt batch
    questions = [problem["question"] for problem in batch]
    prompts = [f"{PAL_PROMPT}\n\nQuestion: {question}\nLet's solve this step by step!\nAnswer:\n" for question in questions]

    # generate completions and extract function that solves problem
    completions = generate_batch(prompts, tokenizer, model, { "max_new_tokens": 200, "do_sample": True, "temperature": 0.3, "top_p": 0.9, "top_k": 50, "num_return_sequences": 1 })
    completions = [pal_process_completion(tokenizer, completion) for completion in completions]

    pal_out = []
    for problem, completion in zip(batch, completions):
        # execute function
        solution, err = pal_execute_completion(completion)
        if err != None:
            # error executing function
            out = PALOut(problem["question"], problem["answer"], completion, None, 0, repr(err))
            pal_out.append(out)
        if solution != problem["answer"]:
            # function returned incorrect solution
            out = PALOut(problem["question"], problem["answer"], completion, solution, 0, None)
            pal_out.append(out)
        else:
            # function returned correct solution
            out = PALOut(problem["question"], problem["answer"], completion, solution, 1, None)
            pal_out.append(out)
    return pal_out

def solve(model, tokenizer, test_data, technique):
    correct = 0

    # split data set into batches of size 10
    batches = split_list(test_data.to_list(), 10)
    for batch in tqdm(batches):
        technique_out = technique(model, tokenizer, batch)
        for out in technique_out:
            correct += out.score
    return correct

def load():
    # load data set
    test_data = datasets.load_dataset("nuprl/engineering-llm-systems", "math_word_problems", split="test")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token 

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16
    ).to(device=DEVICE)

    return test_data, tokenizer, model

def main():
    # load model name
    if len(sys.argv) > 1:
        MODEL = sys.argv[1]

    # load data set, tokenizer, and model
    test_data, tokenizer, model = load()

    # zero shot
    print("Starting Zero Shot Prompting ...")
    zero_shot_correct = solve(model, tokenizer, test_data, zero_shot)
    print(f'Zero Shot Accuracy: {round(zero_shot_correct/50, 2)}')

    # few shot
    print("Starting Few Shot Prompting ...")
    few_shot_correct = solve(model, tokenizer, test_data, few_shot)
    print(f'Few Shot Accuracy: {round(few_shot_correct/50, 2)}')

    # chain of thought
    print("Starting Chain-Of-Thought Prompting ...")
    cot_correct = solve(model, tokenizer, test_data, cot)
    print(f"Chain-of-thought accuracy: {round(cot_correct/50, 2)}")

    # program-aided language model
    print("Starting Program-Aided Language Model Prompting ...")
    pal_correct = solve(model, tokenizer, test_data, pal)
    print(f"Program-Aided Language Models Accuracy: {round(pal_correct/50, 2)}")

    
if __name__ == "__main__":
    main()