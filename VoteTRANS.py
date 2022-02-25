import os
import argparse

from utils.load_model_commands import load_model
from utils.attack_commands import load_attack
from utils.detect_commands import VoteTRANS
from utils.load_dataset_commands import load_dataset_from_huggingface
from utils.data_commands import read_data, save_data
from textattack.attack_results import SkippedAttackResult
from datetime import datetime

from textattack import Attacker
from textattack import AttackArgs

ADV_LABEL = 1
ORG_LABEL = 0
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

def create_arguments():
    """ Create all arguments
    Args:        
    Returns:      
        args (argparse.ArgumentParser):
            all aguments for VoteTRANS

    """
    parser = argparse.ArgumentParser(description='TRUST')
    parser.add_argument('--dataset',
                        help='A dataset',
                        default="ag_news")
    parser.add_argument('--attack',
                        help='An attack',
                        default="pwws")
    parser.add_argument('--target',
                        help='Target model',
                        default="cnn-ag-news")
    parser.add_argument('--auxidiary_attack',
                        help='An auxidiary attack used in VoteTRANS',
                        default="pwws")
    parser.add_argument('--supports',
                        nargs="*", 
                        help='List of support models for VoteTRANS.',
                        default = ['roberta-base-ag-news'])
    parser.add_argument('--num_pairs',
                        help='Number of testing pairs',
                        type=int,
                        default=250)
    parser.add_argument('--word_ratio',
                        help='word_ratio',
                        type=float,
                        default=1.0)
    args = parser.parse_args()
    return args

def print_and_log(text):
    """ print to console and log to text file at "dataset/target/attack/log.txt"    
    Args:        
        text (string):
            text to process
    """  
    print(text)
    with open(f"{args.dataset}/{args.target}/{args.attack}/log.txt", "a+") as f:
        f.write(text + "\n")

def evaluate(org_texts, adv_texts, target, supports, auxidiary_attack, word_ratio = 1.0):    
    """ evaluate VoteTRANS for balanced texts of original and adversarial texts
    Args:        
        org_texts (list of string):
            list of original texts
        adv_texts (list of string):
            list of adversarial texts
        target:
            an target model loaded by TextAttack
        supports:
            support models loaded by TextAttack
        auxidiary_attack:
            an attack from TextAttack.
        word_ratio (float, default = 1.0):
            a threshold to determine the number of words in the input text are used to process        
    Returns:      
        f1 (float):
            F1-score
        recall (float):
            Adverarial recall
    """  
    gold_labels = []
    detect_labels = []
    num_pairs = len(org_texts)
    for index in range(num_pairs):    
#        detect original text
        detect_result = VoteTRANS(org_texts[index], target, supports, auxidiary_attack, word_ratio = word_ratio)            
        gold_labels.append(ORG_LABEL)
        detect_labels.append(detect_result)
        
#        detect adversarial text
        detect_result = VoteTRANS(adv_texts[index], target, supports, auxidiary_attack, word_ratio = word_ratio)            
        gold_labels.append(ADV_LABEL)
        detect_labels.append(detect_result)       
        
#        print results
        org_result = "CORRECT" if (gold_labels[len(gold_labels) - 2] == detect_labels[len(detect_labels) - 2]) else "INCORRECT" 
        adv_result = "CORRECT" if (gold_labels[len(gold_labels) - 1] == detect_labels[len(detect_labels) - 1]) else "INCORRECT"
        
        print_and_log(f"Pair {index + 1} / {len(org_texts)} : Original detection : {org_result} ; Adversarial detection = {adv_result}")
        
    f1 = f1_score(gold_labels, detect_labels, average = 'binary')    
    recall = recall_score(gold_labels, detect_labels, average = 'binary')    
    return f1, recall

args = create_arguments()

def main():
    """ main processing
    """    
    target = load_model(args.target)

    supports = []
    for supporter_name in args.supports:
        supports.append(load_model(supporter_name))

    attack = load_attack(args.attack, target)

    auxidiary_attack = load_attack(args.auxidiary_attack, target)

    if (args.num_pairs != 0):
        if not os.path.exists(f'{args.dataset}/{args.target}/{args.attack}/test-{args.num_pairs}_data.pkl'):

            dataset = load_dataset_from_huggingface(args.dataset)
            attack_args = AttackArgs(num_successful_examples=args.num_pairs)
            attacker = Attacker(attack, dataset, attack_args)
            
            start_attack_time = datetime.now()
            attack_results = attacker.attack_dataset() # attack
            
#            log
            attack_time = datetime.now() - start_attack_time
            if not os.path.exists(f'{args.dataset}/{args.target}/{args.attack}'):
                os.makedirs(f'{args.dataset}/{args.target}/{args.attack}')
            print_and_log("-" * 40 + "Log information" + "-" * 40)            
            print_and_log(f"Log file = {args.dataset}/{args.target}/{args.attack}/log.txt")                        
            print_and_log("-" * 40 + "Attack" + "-" * 40)            
            print_and_log(f'Attack time = {attack_time}')            
            count = 0
            for result in attack_results:
                if not isinstance(result, SkippedAttackResult):
                    count += 1
            print_and_log(f'Attack time per text = {attack_time/count}')           
            save_data(dataset, attack_results, f'{args.dataset}/{args.target}/{args.attack}/test-{args.num_pairs}_data.pkl')
            print_and_log(f'Saved attack data at = {args.dataset}/{args.target}/{args.attack}/test-{args.num_pairs}_data.pkl')
        else:
            print_and_log("-" * 40 + "Log information" + "-" * 40)            
            print_and_log(f"Log file = {args.dataset}/{args.target}/{args.attack}/log.txt")                        
            print_and_log(f'Load attack data at = {args.dataset}/{args.target}/{args.attack}/test-{args.num_pairs}_data.pkl')
        org_texts, adv_texts, _ = read_data(f'{args.dataset}/{args.target}/{args.attack}/test-{args.num_pairs}_data.pkl')        

#        start detection
        print_and_log("-" * 40 + "Detection by VoteTRANS" + "-" * 40)            
        start=datetime.now()
        f1, recall = evaluate(org_texts, adv_texts, target, supports, auxidiary_attack, word_ratio = args.word_ratio) # detect by VoteTRANS
        detection_time = datetime.now()-start
        
#        log
        print_and_log("-" * 40 + "Summarization" + "-" * 40)            
        print_and_log(f"Dataset : {args.dataset}")  
        print_and_log(f"Target : {args.target}")  
        print_and_log(f"Attack for generating adversarial texts : {args.attack}")  
        print_and_log(f"Auxiliary attack for VoteTRANS : {args.auxidiary_attack}")  
        print_and_log(f"Supports for VoteTRANS : {args.supports}")  
        print_and_log(f"Number pairs of original and adversarial texts = {args.num_pairs}")  
        print_and_log(f"F1 score = {f1}")  
        print_and_log(f"Adversarial recall = {recall}")
        print_and_log(f'Detection time = {detection_time}')
        print_and_log(f'Detection time per text = {detection_time / (args.num_pairs * 2)}')
        print_and_log('-' * 80)
    

if __name__ == "__main__":
    main() 