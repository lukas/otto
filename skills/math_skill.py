import simpleeval


class MathSkill:
    function_name = "math"
    parameter_names = ['question']

    examples = [
        [
            "What is two plus two?",
            "math(question=\"2+2\")"
        ],
        [
            "what is three squared plus nine?",
            "math(question=\"3**2+9\")"
        ],
        [
            "tell me a random number below 10",
            "math(question=\"randint(10)\")"
        ]]

    def __init__(self, message_function):
        self.message_function = message_function

    def start(self, args: dict[str, str]):
        if ('question' in args):
            question_string = args["question"]
            answer = simpleeval.simple_eval(question_string)
            self.message_function(answer)


if __name__ == '__main__':
    # for testing
    math = MathSkill(print)
    math.start({"question": "2+2"})
    math.start({"question": "randint(3)"})
