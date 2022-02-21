from textattack.attack_results import SuccessfulAttackResult

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def read_data(file_name):
    with (open(file_name, 'rb')) as inp:
        org_texts, adv_texts, ground_truths = zip(*pickle.load(inp))
    return org_texts, adv_texts, ground_truths

def save_data(dataset, attack_results, file_name):
    org_texts = []
    adv_texts = []
    ground_truths = []
    for i, result in enumerate(attack_results):
        if isinstance(result, SuccessfulAttackResult):
            __, ground_truth = dataset[i]
            org_text = result.original_text()
            adv_text = result.perturbed_text()
            org_texts.append(org_text)
            adv_texts.append(adv_text)
            ground_truths.append(ground_truth)
    save_object(zip(org_texts,adv_texts, ground_truths), file_name)            