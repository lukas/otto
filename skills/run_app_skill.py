import subprocess


class RunAppSkill:
    function_name = "runapp"
    parameter_names = ['program']

    examples = [
        [
            "Open chrome",
            "runapp(program=\"Google Chrome\")"
        ],
        [
            "run safari",
            "runapp(program=\"Safari\")"
        ],
        [
            "start vscode"
            "runapp(program=\"Visual Studio Code\")"
        ]]

    def __init__(self, message_function):
        self.message_function = message_function

    def start(self, args: dict[str, str]):
        if ('program' in args):
            program_string = args["program"]
            subprocess.run(["open", "-a", program_string])

            self.message_function("ok")


if __name__ == '__main__':
    # for testing
    math = RunAppSkill(print)
    math.start({"program": "safari"})
