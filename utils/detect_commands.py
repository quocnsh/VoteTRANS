import torch
import math
import numpy as np
from textattack.shared import  AttackedText
import inspect
ADV_LABEL = 1
ORG_LABEL = 0


def predict(model, text):
    """ predict a text using a model
    Args:        
        model (textattack.models.wrappers.ModelWrapper):
            a model loaded by TextAttack
        text (str):
            a text
    Returns:      
        distribution (numpy array):
            distribution prediction for the input text
    """   
    return predicts(model, [text])[0]

def get_candidate_indexes(pre_transformation_constraints, original_text, transformation):
    """ checking the input text (original text) with pre-constraints
    Args:
        pre_transformation_constraints: 
            list of pre-constrainst
        original_text:
            the original text
        transformation:
            transformation is used for checking
    Returns:
        indexes(array):
            list of indexes statified the pre-constrainst
    """
    indexes = None    
    for constrainst in pre_transformation_constraints:
        if indexes == None:
            indexes = set(constrainst(original_text, transformation))        
        else:
            indexes = indexes.intersection(constrainst(original_text, transformation))        
    return set(indexes)


def get_index_priority(search_method, initial_text, selected_indexes):
    """ sort the selected_indexes by priority
    Args:
        search_method:
            the method determines the priority
        initial_text:
            the original text
        selected_indexes:
            indexes are statisfied the pre-constrainsts
        
    Returns:
        indexes(array):
            list of indexes after sorting
    """    
    if (hasattr(search_method, '_get_index_order') and inspect.isfunction(search_method._get_index_order)):
        index_order, search_over = search_method._get_index_order(initial_text)
        indexes = []
        for index in index_order:
            if index in selected_indexes:
               indexes.append(index) 
        return indexes
    else:
        return selected_indexes


def priority(attack, input_text):
    """ determine the priority for defense
    Args:
        attack (textattack.Attack):
            an attack from TextAttack
        input_text:
            the original text
    Returns:
        indexes(array):
            list of indexes after priority
    """    
    original_text = AttackedText(input_text)
    indexes = get_candidate_indexes(attack.pre_transformation_constraints, original_text, attack.transformation)  
    indexes = get_index_priority(attack.search_method, original_text, list(indexes))
    return indexes


def transform(transformation, input_text, word_index):
    """ transform a word at the word_index in input_text
    Args:
        transformation:
            transformation of an attack from TextAttack
        input_text:
            the original text
        word_index:
            index of word to modify
    Returns:
        transform_texts(array):
            list of texts after transformation
    """ 
    original_text = AttackedText(input_text)
    transform_texts = transformation(original_text, indices_to_modify = [word_index])
    return transform_texts

def constraint(transformed_texts, attack, input_text):
    """ check constrainsts for transformed texts
    Args:        
        transformed_texts:
            transformed texts
        attack:
            An attack from TextAttack framework
        input_text:
            an input text
    Returns:                
        statisfied_texts: 
            all texts in the transformed texts statify the constrainsts
    """ 
    original_text = AttackedText(input_text)
    filtered_texts = attack.filter_transformations(transformed_texts, original_text, original_text)
    result = []
    for filter_text in filtered_texts:
        result.append(filter_text.text.strip())
    return result
    

def predicts(model_predict, inputs, batch_size=32):

    """ Runs prediction on iterable ``inputs``
    Input:
        model_predict: 
            a model from TextAttack framework
        inputs:
            list of texts
    Output:
        all predictions into an ``np.ndarray``
    """
    outputs = []
    i = 0
    while i < len(inputs):
        batch = inputs[i : i + batch_size]
        batch_preds = model_predict(batch)

        # Some seq-to-seq models will return a single string as a prediction
        # for a single-string list. Wrap these in a list.
        if isinstance(batch_preds, str):
            batch_preds = [batch_preds]

        # Get PyTorch tensors off of other devices.
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        # Cast all predictions iterables to ``np.ndarray`` types.
        if not isinstance(batch_preds, np.ndarray):
            batch_preds = np.array(batch_preds)
        outputs.append(batch_preds)
        i += batch_size

    return np.concatenate(outputs, axis=0)


def majority_vote(distributions):
    """ majority_vote (hard voting)
    Input:
        distributions: list of distributions
    Output:
        indexes: get indexes of main class from distributions
    """
    num_class = distributions.shape[1]
    count = np.zeros((num_class))
    max_indexes = np.argmax(distributions, axis = 1)
    for value in max_indexes:
        count[value] += 1
        
    winner = np.argwhere(count == np.amax(count))        
    indexes = winner.flatten().tolist()
    return indexes

def VoteTRANS(input_text, target, supporters, attack,  word_ratio = 1.0):
    """ VoteTRANS for detecting adversarial text (Algorithm 1 in the paper)
    Args:        
        input_text (X):
            the input text 
        target (F):
            an target model loaded by TextAttack
        supports (F^{sup}):
            support models loaded by TextAttack
        attack (A^{imp}, A^{trans}, A^{cons}):
            an attack from TextAttack. It includes word importance A^{imp}, word transformation A^{trans}, and constrainst A^{cons}
        word_ratio (\alpha, float, default = 1.0):
            a threshold to determine the number of words in the input text are used to process        
    Returns:
        Adversarial text detection (ADV_LABEL/ORG_LABEL):
    """
    predict_index = np.argmax(predict(target, input_text))     
    word_indexes = priority(attack, input_text) # sorted in descending order of the importance score
    
    num_word = int(math.floor(len(word_indexes) * word_ratio)) 
    word_indexes = word_indexes[:num_word]# get top word_ratio words
    
    for word_index in word_indexes:
        
        valid_texts = []
        
        transformed_texts = transform(attack.transformation, input_text, word_index) # create transformation
        
        filtered_texts = constraint(transformed_texts, attack, input_text) # check constraint        
        
        for transformed_text in filtered_texts:
            valid_texts.append(transformed_text) 
            
        if (len(valid_texts) > 0):
            merge_predictions = predicts(target, valid_texts) # Y^{trans}
            
            for supporter in supporters:  
                support_predictions = predicts(supporter, valid_texts)
                merge_predictions = np.concatenate((merge_predictions, support_predictions), axis=0)
                
            majority_indices = set(majority_vote(merge_predictions)) # Y'
            
            if not (predict_index in majority_indices) or (predict_index in majority_indices and len(majority_indices) > 1):
                return ADV_LABEL        
            
    return ORG_LABEL
