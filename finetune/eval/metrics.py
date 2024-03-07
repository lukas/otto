import weave
import re

@weave.op()
def match(example: dict, prediction: dict) -> dict:
    scores = {}
    scores['acc'] = prediction['generated_text'] == example['answer']
    # check if prediction matches a function call format like op(a="b") using regular expression
    pattern = r'\w+\((\w+="[^"]*",\s*)*(\w+="[^"]*")?\s*\)'
    match = re.match(pattern, prediction['generated_text'])
    scores['correct format'] = bool(match)
    
    # get the name of the function of the example answer
    pattern = r'([^(]+)\(.*'
    match = re.match(pattern, example['answer'])
    function_name = match.group(1)
    scores[function_name + ' acc'] = prediction['generated_text'] == example['answer']


    return scores



@weave.op()
def example_to_model_input(example:dict) -> str:
    return example