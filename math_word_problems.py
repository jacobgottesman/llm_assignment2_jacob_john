import datasets
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import json 
import torch
from collections import namedtuple
from pprint import pprint
from tqdm.auto import tqdm

PALOut = namedtuple("PALOut", ["question", "answer", "model_completion", "model_answer", "score", "error"])

MODEL = "/scratch/bchk/aguha/models/llama3p1_8b_base"
DEVICE = "cuda"

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

Question: Sally has 3 bags of marbles. The first bag has 12 marbles, the second has 8 marbles, and she has 5 fewer marbles in the third bag than the first bag. How many marbles does Sally have in total?
Let's solve this step by step!
Answer:
def solve():
    marbles_in_first_bag = 12
    marbles_in_second_bag = 8
    marbles_in_third_bag = marbles_in_first_bag - 5
    total_marbles = marbles_in_first_bag + marbles_in_second_bag + marbles_in_third_bag
    return total_marbles

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

Question: A restaurant has 45 tables. If 28 tables are occupied and each table seats 4 people, how many empty seats are there?
Let's solve this step by step!
Answer:
def solve():
    total_tables = 45
    seats_per_table = 4
    total_seats = total_tables * seats_per_table
    occupied_tables = 28
    occupied_seats = occupied_tables * seats_per_table
    empty_seats = total_seats - occupied_seats
    return empty_seats
    
Question: Alex runs 4 laps during each workout session, and he works out 4 times a week. Each lap is 70 meters long. How many total meters does Alex run in a week?
Let's solve this step by step!
Answer:
def solve():
    laps_per_session = 4
    sessions_per_week = 4
    laps_per_week = laps_per_session * sessions_per_week
    meters_per_lap = 70
    total_meters = meters_per_lap * laps_per_week
    return total_meters

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
    
Question: Ben eats one sandwich a day and buys packs that contain 10 sandwiches each at a cost of $5 per pack. How much will he spend on sandwiches in 50 days?
Let's solve this step by step!
Answer:
def solve():
    sandwiches_per_day = 1
    num_days = 50
    total_sandwiches = sandwiches_per_day * num_days
    sandwiches_per_pack = 10
    num_packs_needed = total_sandwiches / sandwiches_per_pack
    cost_per_pack = 5
    total_cost = num_packs_needed * cost_per_pack
    return total_cost

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

def generate_batch(prompts, tokenizer, model):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            top_k=50,
            num_return_sequences=1)
    return [tokenizer.decode(output[inputs["input_ids"].shape[1]:]) for output in outputs]

def execute_completion(completion):
    try:
        exec(completion)
        ans = round(eval("solve()"))
        return ans, None
    except Exception as e:
        return None, e

def process_completion(tokenizer, completion):
    split_index = min(
        completion.find("Question") if "Question" in completion else float("inf"),
        completion.find(tokenizer.eos_token) if tokenizer.eos_token in completion else float("inf")
    )
    return completion[:split_index].strip() if split_index != float("inf") else completion.strip()

def pal_batch(model, tokenizer, batch):
    questions = [problem["question"] for problem in batch]
    prompts = [f"{PAL_PROMPT}\n\nQuestion: {question}\nLet's solve this step by step!\nAnswer:\n" for question in questions]
    completions = generate_batch(prompts, tokenizer, model)
    completions = [process_completion(tokenizer, completion) for completion in completions]
    pal_batch_out = []
    for problem, completion in zip(batch, completions):
        model_answer, err = execute_completion(completion)
        if err != None:
            pal_out = PALOut(problem["question"], problem["answer"], completion, None, 0, repr(err))
            pal_batch_out.append(pal_out)
            pprint(pal_out._asdict(), indent=2)
        if model_answer != problem["answer"]:
            pal_out = PALOut(problem["question"], problem["answer"], completion, model_answer, 0, None)
            pal_batch_out.append(pal_out)
            pprint(pal_out._asdict(), indent=2)
        else:
            pal_out = PALOut(problem["question"], problem["answer"], completion, model_answer, 1, None)
            pal_batch_out.append(pal_out)
    return pal_batch_out

def split_list(dict_list, x):
    return [dict_list[i:i + x] for i in range(0, len(dict_list), x)]

def pal(model, tokenizer, test_data):
    correct = 0
    batches = split_list(test_data.to_list(), 10)
    for batch in tqdm(batches):
        pal_batch_out = pal_batch(model, tokenizer, batch)
        for pal_out in pal_batch_out:
            correct += pal_out.score
    return correct

def main():

    # load in data
    test_data = datasets.load_dataset("nuprl/engineering-llm-systems", "math_word_problems", split="test")

    n_samp = 5

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

    num_correct = pal(model, tokenizer, test_data)
    print(f"Program-Aided Language Models Accuracy: {round(num_correct/len(test_data), 2)}")

    
if __name__ == "__main__":
    main()


