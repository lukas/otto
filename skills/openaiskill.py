from dotenv import load_dotenv
import openai
from openai import ChatCompletion


load_dotenv()


class OpenAISkill:
    function_name = "openai"
    parameter_names = ['prompt']
    examples = [
        [
            "Ask openai if whales are mammals",
            "openai(prompt=\"Are whales mammals?\")"
        ],
        [
            "Ask gpt if a leopard can swim",
            "openai(prompt=\"Can a leopard swim?\")"
        ],
        [
            ["Ask open ai for an interesting fact about the natural world that a smart educated person might not know"],
            ["openai(prompt=\"Tell me an interesting fact about the natural world that a smart educated person might not know\")"]
        ]
    ]

    def __init__(self, message_function):
        self.message_function = message_function

    def start(self, args: dict[str, str]):
        question = args["prompt"]
        if (question != None and question != ""):
            s = self._ask_gpt3(question)

    def _ask_gpt3(self, question: str):
        response = ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]
        )
        answer = response["choices"][0]["message"]["content"]
        self.message_function(answer)


if __name__ == '__main__':
    # for testing
    openai = OpenAISkill(print)
    openai.start({"prompt":
                  "Tell me an interesting fact about the natural world that a smart educated person might not know"})
