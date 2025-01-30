from datasets import load_dataset
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import json 

from timeout_decorator import timeout, TimeoutError

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
    pattern = "|".join(["\ndef", "\nclass", "\nif", "\nprint"])
    clipped_solutions = [re.split(pattern, solution)[0] for solution in solutions]

    return clipped_solutions

def test_solutions(sols, problem):
    results = []
    failures = {}
    for i, sol in enumerate(sols.values()):
        try:
            code = str(problem['prompt'] + '\n' + sol + '\n' + problem["tests"])
            execute_code(code)  # Executes with a 5-second timeout
            results.append(1)
        except TimeoutError:
            failures[i] = {"solution": sol, "error": "Execution timed out"}
            results.append(0)
        except Exception as e:
            failures[i] = {"solution": sol, "error": str(e)}
            results.append(0)
    return results, failures