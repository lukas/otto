import weave
from weave import weaveflow
from ..ft_utils import load_test_ds

def publish_eval_datasets():
    test_dataset = load_test_ds(args)

    test_dataset_list_of_dict = weaveflow.Dataset(test_dataset.to_pandas().to_dict('records'))
    small_test_dataset = weaveflow.Dataset(test_dataset.to_pandas()[:5].to_dict('records'))
    weave.publish(test_dataset_list_of_dict, 'test-labels')
    weave.publish(small_test_dataset, 'test-labels-small')

    dataset = weave.ref("test-labels").get()