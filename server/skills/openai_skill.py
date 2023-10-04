from skills.base import Skill
from dotenv import load_dotenv
import openai
from openai import ChatCompletion
import os

load_dotenv()


class OpenAISkill(Skill):
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
            "Ask open ai for an interesting fact about the natural world that a smart educated person might not know",
            "openai(prompt=\"Tell me an interesting fact about the natural world that a smart educated person might not know\")"
        ]
    ]

    def start(self, args: dict[str, str]):
        if 'prompt' in args:
            question = args["prompt"]
        else:
            question = ""
            print("No prompt specified!")
            return

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
    openai.api_base = "https://proxy.wandb.ai/proxy/openai/v1"
    old_api_key = os.environ["OPENAI_API_KEY"]
    wandb_key = os.environ["WANDB_API_KEY"]
    openai.api_key = f"{wandb_key}:{old_api_key}"
    print("API Key", openai.api_key)
    # os.environ["OPENAI_API_BASE"] = "https://api.wandb.ai/proxy/openai/v1"
    # os.environ["OPENAI_API_KEY"] = os.environ["WANDB_API_KEY"] + \
    #     ":" + os.environ["OPENAI_API_KEY"]
    # for testing

    openai = OpenAISkill(print)
    openai.start({"prompt":
                  "Tell me an interesting fact about the natural world that a smart educated person might not know"})
