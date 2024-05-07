import argparse
import weave
from types import SimpleNamespace

from ft_utils import (
    load_model_from_artifact, 
    load_ds_from_artifact, 
    create_mistral_instruct_prompt, 
    create_llama_prompt,
    create_llama_chat_prompt,
    create_custom_prompt,
    load_model_from_hf,
    read_file,
)

from eval.anthropic_eval import evaluate_anthropic
from eval.openai_eval import evaluate_openai
from eval.finetune_eval import evaluate_model

def validate_prompt_format(prompt):
    if not ("user" in prompt and "answer" in prompt):
        raise Exception("The prompt is not formatted correctly, you need to provide: user, answer")
    else:
        return create_custom_prompt(prompt)

def get_default_create_prompt(args):
    "This should come from the model artifact metadata"
    if "mistral" in args.MODEL_AT.lower():
        create_prompt = create_mistral_instruct_prompt
    elif "llama" in args.MODEL_AT.lower():
        if "chat" in args.MODEL_AT.lower():
            create_prompt = create_llama_chat_prompt
        else:
            create_prompt = create_llama_prompt
    else:
        raise Exception("Model not recognized")
    return create_prompt



defaults = SimpleNamespace(
    MODEL_AT = 'capecape/huggingface/6urzaw17-mistralai_Mistral-7B-Instruct-v0.1-ft:v0',
    DATASET_AT = 'capecape/otto/split_dataset:v2',
    MODEL_ID = None,
    PROMPT = None,
)

def parse_args(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--publish", action="store_true")

    # parser.add_argument("--MODEL_AT", type=str, default=defaults.MODEL_AT)
    # parser.add_argument("--DATASET_AT", type=str, default=defaults.DATASET_AT)
    # parser.add_argument("--MODEL_ID", type=str, default=defaults.MODEL_ID)
    # parser.add_argument("--PROMPT", type=str, default=defaults.PROMPT)
    # parser.add_argument("--PROMPT_FILE", type=str, default=None)
    return parser.parse_args()

def load_test_ds(dataset):
    # if args.PROMPT is not None: # custom prompt
    #     create_prompt = validate_prompt_format(args.PROMPT)
    # else:
    #     create_prompt = get_default_create_prompt(args)

    # # to parse the hf dataset
    # create_test_prompt = lambda row: {"text": create_prompt({"user":row["user"], "answer":""})}
    
    ds = load_ds_from_artifact(dataset)
    test_dataset = ds["test"]
    return test_dataset

def publish_eval_datasets(dataset_ref: str):
    test_dataset = load_test_ds(dataset_ref)

    test_dataset_list_of_dict = weave.Dataset(rows=test_dataset.to_pandas().to_dict('records'), name='test-labels')
    small_test_dataset = weave.Dataset(rows=test_dataset.to_pandas()[:5].to_dict('records'), name='test-labels-small')
    weave.publish(test_dataset_list_of_dict)
    weave.publish(small_test_dataset)

    dataset = weave.ref("test-labels").get()




# def evaluate(args):
#     # initialize run
#     wandb.init(project="otto", job_type="eval", config=args)

#     test_dataset = load_test_ds(args)
#     model, tokenizer = load_model(args)

#     table, acc, acc_lousy = create_predictions_table(model, tokenizer, test_dataset, 64)

#     wandb.log({"eval_predictions":table})
#     wandb.run.summary["acc"] = acc
#     wandb.run.summary["acc_lousy"] = acc_lousy
#     wandb.finish()
    
if __name__ == "__main__":

    weave.init("otto11")

    args = parse_args(defaults)

    if args.publish:
        publish_eval_datasets('capecape/otto/split_dataset:v2')
        exit()

    if args.dataset == 'small':
        eval_dataset_name = 'test-labels-small'
    elif args.dataset == 'test':
        eval_dataset_name = 'test-labels'
    elif args.dataset == 'synthetic':
        eval_dataset_name = 'synthetic-data'
    elif args.dataset == 'synthetic-small':
        eval_dataset_name = 'synthetic-data-small'


    # publish_eval_datasets('capecape/otto/split_dataset:v2')

    # if args.PROMPT_FILE is not None:
    #     print(f"Reading prompt from file: {args.PROMPT_FILE}")
    #     args.PROMPT = read_file(args.PROMPT_FILE)
    if args.model == "anthropic":
        evaluate_anthropic(eval_dataset_name)
    elif args.model == "openai":
        evaluate_openai(eval_dataset_name)
    elif args.model == "custom":
        evaluate_model(eval_dataset_name)
    # else:
    #    evaluate_anthropic_bt(eval_dataset_name)