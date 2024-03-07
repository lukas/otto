import weave
from weave import weaveflow
from openai import OpenAI
from ft_utils import (
    read_file
)
import asyncio

from .metrics import match, example_to_model_input



@weave.type()
class GenTextOpenAI(weaveflow.Model):
    model_name: str
    system_prompt: str
    user_template: str

    @weave.op()
    async def predict(self, example: dict) -> dict:
        # prompt = example["text"]
        user_query = example["user"]
        label = example["answer"]
        model_client = OpenAI(
            # This is the default and can be omitted
            # api_key=os.environ["OPENAI_API_KEY"]
        )

        response = model_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query},
            ],
            max_tokens=100
        )
        print(response.choices[0].message.content)
        return {'generated_text': response.choices[0].message.content}
    
def evaluate_openai(dataset_name: str):
    dataset = weave.ref(dataset_name).get()

    openai_system_prompt = read_file("prompts/openai_system.txt")
    openai_user_tempate = read_file("prompts/openai_user.txt")



    weave_model = GenTextOpenAI("gpt-4", openai_system_prompt, openai_user_tempate)
    eval = weaveflow.Evaluation(dataset, scores=[match], example_to_model_input=example_to_model_input)
    asyncio.run(eval.evaluate(weave_model))
    # table, acc, acc_lousy = create_predictions_table(model, tokenizer, test_dataset, 64)