import weave
import ast


@weave.op()
def match(answer: str, prediction: dict ) -> dict:
    scores = {}

    pred_str = prediction['generated_text']

    scores['acc'] = function_calls_match(answer, pred_str)

    scores['correct format'] = is_valid_function(pred_str)
    

    function_name = function_call_name(answer)
    scores[function_name + ' acc'] = scores['acc'] 


    return scores


def function_call_name(f) -> str:
    try:
        # Parse the function calls into AST nodes
        f1_ast = ast.parse(f.strip()).body[0].value
        return f1_ast.func.id 
    except Exception as e:
        return None

def is_valid_function(f) -> bool:
    try:
        # Parse the function calls into AST nodes
        f1_ast = ast.parse(f.strip()).body[0].value
        return True
    except Exception as e:
        return False
    
def function_calls_match(f1: str, f2: str) -> bool:
    # function call in the format of a(b="c", d='f')
    # see if function, args and values are all the same between f1 and f2

    try:
        # Parse the function calls into AST nodes
        f1_ast = ast.parse(f1.strip()).body[0].value
        f2_ast = ast.parse(f2.strip()).body[0].value

        # Check if the function names are the same
        if f1_ast.func.id != f2_ast.func.id:
            return False

        # Check if the number of arguments is the same
        if len(f1_ast.args) != len(f2_ast.args) or len(f1_ast.keywords) != len(f2_ast.keywords):
            return False

        # Check if all positional arguments are the same
        for arg1, arg2 in zip(f1_ast.args, f2_ast.args):
            if ast.dump(arg1) != ast.dump(arg2):
                return False

        # Check if all keyword arguments are the same
        for kw1, kw2 in zip(f1_ast.keywords, f2_ast.keywords):
            if kw1.arg != kw2.arg or ast.dump(kw1.value) != ast.dump(kw2.value):
                return False

        return True
    except Exception as e:
        # If there's an error in parsing, assume the function calls do not match
        return False


