import anthropic
import weave
from weave import weaveflow

import asyncio

from .metrics import match

class GenTextAnthropic(weave.Model):
    model_name: str
    system_prompt: str
    user_template: str

    @weave.op()
    async def predict(self, user:str) -> dict:
        # prompt = example["text"]
        user_query = user
        client = anthropic.Anthropic(

        )

        message = client.messages.create(
            model=self.model_name,
            system=self.system_prompt,
            max_tokens=1000,
            temperature=0,
            messages=[
              {"role": "user",
                "content": [
                {
                    "type": "text",
                    "text": user_query
                }
                ]} 
            ]
        )
        content_block = message.content[0]
        print(content_block.text)
      
        return {'generated_text': message.content[0].text}
    

def evaluate_anthropic(dataset_name: str):
    dataset = weave.ref(dataset_name).get()
    
    openai_system_prompt = open("prompts/openai_system.txt", 'r').read()
    openai_user_tempate = open("prompts/openai_user.txt", "r").read()

    # dataset_ref = weave.publish(small_test_dataset, 'test-labels-small')

    weave_model = GenTextAnthropic(model_name="claude-3-sonnet-20240229", 
                                    system_prompt=openai_system_prompt, 
                                    user_template=openai_user_tempate)
    eval = weave.Evaluation(dataset=dataset, scores=[match])
    eval = weave.Evaluation(dataset=dataset, scores=[match])
    asyncio.run(eval.evaluate(weave_model))
    # table, acc, acc_lousy = create_predictions_table(model, tokenizer, test_dataset, 64)