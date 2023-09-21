from dotenv import load_dotenv
import openai
from openai import ChatCompletion

load_dotenv()


class StorySkill:
    function_name = "story"
    parameter_names = ['description']

    examples = [
        [
            "Tell me a story"
            "story()"
        ],
        [
            "Tell me a story about a cat and a dog",
            "story(description=\"cat and dog\")"
        ],
        [
            "Make up a story about two best friends that fly to the moon",
            "story(description=\"two best friends that fly to the moon\")"
        ]]

    def __init__(self, message_function):
        self.message_function = message_function

    def start(self, args: dict[str, str]):
        prompt = "Tell me a story"
        if ('description' in args):
            prompt += args['description']

        s = self._ask_gpt3(prompt)

    # Could also do this with llama
    def _ask_gpt3_story(self, question: str):
        response = ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a story telling assistant."},
                {"role": "user", "content": question},
            ]
        )
        answer = response["choices"][0]["message"]["content"]
        self.message_function(answer)


if __name__ == '__main__':
    # for testing
    math = MathSkill(print)
    math.start({"question": "2+2"})
    math.start({"question": "randint(3)"})
