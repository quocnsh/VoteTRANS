# VoteTRANS: Detecting Adversarial Text without Training by Voting on Hard Labels of Transformations

## Dependencies

* TextAttack framework (https://github.com/QData/TextAttack) (install by : `pip install textattack` for basic packages (such as PyTorch and Transformers) or `pip install textattack [tensorflow,optional]` for tensorflow and optional dependences. Please refer TextAttack installation (https://textattack.readthedocs.io/en/latest/0_get_started/installation.html) for more detail.

## Usage

* Run `python3 VoteTRANS.py`

### Parameters

* `dataset` : dataset name (default = 'ag_news'), complied with the names from HuggingFaceDataset (https://github.com/huggingface/datasets/tree/master/datasets). Other names are used
 in the paper including  'imdb', 'mr' and 'yelp' for IMDB, Rotten Tomatoes Movie Review, and Yelp Polarity, respectively.
* `attack`: attack name (default = 'pwws'), compied with the names from TextAttack. Other attack names can be found in the TextAttack framework (https://textattack.readthedocs.io/en/latest/3recipes/attack_recipes_cmd.html#attacks-and-papers-implemented-attack-recipes-textattack-attack-recipe-recipe-name)
* `target`: name of target model (default = 'cnn-ag-news'), complied with the names from TextAttack. Other model name can be found in the TextAttack framework (https://textattack.readthedocs.io/en/latest/3recipes/models.html#textattack-models)
* `auxidiary_attack`: an audixiary attack name (default = 'pwws')
* `supports`: list of support-model names (default = ['roberta-base-ag-news'])
* `num_pairs`:  number of testing pairs (default = 250)
* `word_ratio`:  word ratio (default = 1.0) 

### Example

* Running with default parameters : `python3 VoteTRANS.py`
* Running with customized parameters : `python3 VoteTRANS.py --dataset imdb --attack pwws --target cnn-imdb --auxidiary_attack textfooler --supports lstm-imdb bert-base-uncased-imdb --num_pairs 5 --word_ratio 0.1`

### Acknowledgement
* TextAttack framework (https://github.com/QData/TextAttack)
