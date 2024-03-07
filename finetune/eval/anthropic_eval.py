import anthropic
import weave
from weave import weaveflow
from ft_utils import (
    read_file
)
import asyncio

from .metrics import match, example_to_model_input

@weave.type()
class GenTextAnthropic(weaveflow.Model):
    model_name: str
    system_prompt: str
    user_template: str

    @weave.op()
    async def predict(self, example: dict) -> dict:
        # prompt = example["text"]
        user_query = example["user"]
        label = example["answer"]
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
    
    openai_system_prompt = read_file("prompts/openai_system.txt")
    openai_user_tempate = read_file("prompts/openai_user.txt")


    # dataset_ref = weave.publish(small_test_dataset, 'test-labels-small')

    weave_model = GenTextAnthropic("claude-3-sonnet-20240229", openai_system_prompt, openai_user_tempate)
    eval = weaveflow.Evaluation(dataset, scores=[match], example_to_model_input=example_to_model_input)
    asyncio.run(eval.evaluate(weave_model))
    # table, acc, acc_lousy = create_predictions_table(model, tokenizer, test_dataset, 64)