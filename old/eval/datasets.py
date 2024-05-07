import weave
from weave import weaveflow
from ft_utils import load_test_ds

def publish_eval_datasets():
    test_dataset = load_test_ds()

    test_dataset_list_of_dict = weave.Dataset(rows=test_dataset.to_pandas().to_dict('records'), name='test-labels')
    small_test_dataset = weave.Dataset(rows=test_dataset.to_pandas()[:5].to_dict('records'), name='test-labels-small')
    weave.publish(test_dataset_list_of_dict)
    weave.publish(small_test_dataset)

    dataset = weave.ref("test-labels").get()