import typing
import functools
import weave

from weaveflow import base_types
from weaveflow import cli


# TODO: optional
class HFModelConfig(typing.TypedDict):
    model_name: str


# Model
@weave.type()
class HFModel(base_types.Model):
    config: HFModelConfig

    @weave.op()
    def predict(self, user: str) -> str:
        import functools
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        @functools.cache
        def tokenizer_model():
            model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                # quantization_config=bnb_config,
                # device_map="auto", # dispatch efficiently the model on the available ressources
                # max_memory = {i: max_memory for i in range(n_gpus)},
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_name"], use_auth_token=True
            )
            return tokenizer, model

        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        INSTRUCTION_KEY = "### User:"
        RESPONSE_KEY = "### Answer:"
        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY} {user}"
        response = f"{RESPONSE_KEY} "
        formatted_prompt = f"{blurb}\n{instruction}\n{response}\n"
        tokenizer, model = tokenizer_model()
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        output = model.generate(**inputs)
        return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    cli.weave_object_main(HFModel)
