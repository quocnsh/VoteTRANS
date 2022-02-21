import textattack

TEST_SET_ARGS = {
        "sst2": ("glue","sst2","validation"),
        "imdb": ("imdb",None,"test"),
        "ag_news": ("ag_news",None,"test"),
        "cola": ("glue", "cola", "validation"),
        "mrpc": ("glue", "mrpc", "validation"),
        "qnli": ("glue", "qnli", "validation"),
        "rte": ("glue", "rte", "validation"),
        "wnli": ("glue", "wnli", "validation"),
        "mr": ("rotten_tomatoes", None, "test"),
        "snli": ("snli", None, "test"),
        "yelp": ("yelp_polarity", None, "test"),
        }

def load_dataset_from_huggingface(dataset_name):
    if dataset_name in TEST_SET_ARGS:
        dataset = textattack.datasets.HuggingFaceDataset(
            *TEST_SET_ARGS[dataset_name], shuffle=False
        )
    else:
        dataset = textattack.datasets.HuggingFaceDataset(
            *(dataset_name, None, "test"), shuffle=False
        )
    return dataset