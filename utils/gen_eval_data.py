import weave
from openai import OpenAI

import asyncio
import json

class GenEvalDataOpenAI(weave.Model):
    model_name: str
    system_prompt: str
    user_prompt: str

    @weave.op()
    async def predict(self) -> dict:
        # prompt = example["text"]

        model_client = OpenAI(
            # This is the default and can be omitted
            # api_key=os.environ["OPENAI_API_KEY"]
        )

        response = model_client.chat.completions.create(
            model=self.model_name,
            # response_format={ "type": "json_object" },
            messages=[
                # {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content

def publish_json(filename: str, name:str):

    data = json.load(open(filename, 'r'))
    updated_data = [{'user': item['input'], 'answer': item['output']} for item in data]

    dataset = weave.Dataset(rows=updated_data, name=name)

    dataset_small = weave.Dataset(rows=updated_data[:5], name=name+'-small')
    weave.publish(dataset)
    weave.publish(dataset_small)



if __name__ == "__main__":
    weave.init("otto8")
    # system_prompt = open("prompts/gen_eval_data_system.txt", 'r').read()
    # user_prompt = open("prompts/gen_eval_data_user.txt", 'r').read()
    # weave_model = GenEvalDataOpenAI(model_name="gpt-4", user_prompt=user_prompt, system_prompt=system_prompt, name="gpt-4-no-system")
    
    # for i in range(2):
    #     output = asyncio.run(weave_model.predict())
    #     try:
    #         data = json.loads(output)
    #     except json.JSONDecodeError:
    #         print("Received non-JSON response:", output)
    #         # Handle the error as needed, e.g., by setting data to None or logging the error
    #         data = None

    #     # write data to a json file
    #     with open("validation_data.json", "w") as f:
    #         json.dump(data, f, indent=4)

    #     print("Wrote data to validation_data.json")
    
    # mall_test_dataset = weave.Dataset(rows=test_dataset.to_pandas()[:5].to_dict('records'), name='test-labels-small')
    # weave.publish(test_dataset_list_of_dict)
    publish_json('validation_data.json', 'synthetic-data')
    
