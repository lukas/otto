import weave

from openai import OpenAI

import asyncio

from .metrics import match



class GenTextOpenAIModel(weave.Model):
    model_name: str
    system_prompt: str
    user_template: str

    @weave.op()
    async def predict(self, user: str) -> dict:
        # prompt = example["text"]

        model_client = OpenAI(
            # This is the default and can be omitted
            # api_key=os.environ["OPENAI_API_KEY"]
        )

        response = model_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
            max_tokens=100
        )
        print(response.choices[0].message.content)
        return {'generated_text': response.choices[0].message.content}
    
def evaluate_openai(dataset_name: str):
    dataset = weave.ref(dataset_name).get()

    openai_system_prompt = open("prompts/openai_system.txt", 'r').read()
    openai_user_tempate = open("prompts/openai_user.txt", "r").read()

    weave_model = GenTextOpenAIModel(model_name="gpt-4", system_prompt=openai_system_prompt, user_template=openai_user_tempate, name="gpt-4")
    eval = weave.Evaluation(dataset=dataset, scores=[match])
    asyncio.run(eval.evaluate(weave_model))
    # table, acc, acc_lousy = create_predictions_table(model, tokenizer, test_dataset, 64)