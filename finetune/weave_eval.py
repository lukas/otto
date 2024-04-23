from dataclasses import dataclass
import simple_parsing
import weave

import torch
import datasets
from transformers import pipeline

from ft_utils import load_ds_from_artifact, llama_prompt

@dataclass
class Args:
    model_id: str = 'capecape/huggingface/6urzaw17-mistralai_Mistral-7B-Instruct-v0.1-ft:v0'
    dataset_at: str = 'capecape/otto/split_dataset:v2'



dataset = datasets.load_from_disk("/Users/tcapelle/work/otto/finetune/artifacts/split_dataset:v2/test")
# dataset = load_ds_from_artifact(args.dataset_at)

def hf_to_weave(dataset: datasets.Dataset) -> weave.Dataset:
    list_ds = dataset.to_list()
    return weave.Dataset(rows=list_ds, name='test-ds')


class HFModel(weave.Model):
    model_name: str = 'llama2'
    system_prompt: str = llama_prompt
    max_new_tokens: int = 64
    temperature: float = 0.7

    def __post_init__(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    @weave.op()
    def generate(self, user_prompt: str) -> str:
        messages = [self.system_prompt.format(user=user_prompt)]
        outputs = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
        )
        return outputs[0]["generated_text"][-1]["content"]


@weave.op()
def match(answer: str, prediction: dict ) -> dict:
    "a row -> {'user': 'Cheers!', 'answer': 'other()'}"
    return {
        "acc": answer.strip() == prediction.strip(),
        "acc_lousy": answer.strip().lower() == prediction.strip().lower()
        }


if __name__ == "__main__":
    args = simple_parsing.parse(Args)

    wds = hf_to_weave(dataset)
    print(wds)

    model = HFModel()

    eval = weave.Evaluation(dataset=wds, scorers=[match])
    eval.evaluate(model)
