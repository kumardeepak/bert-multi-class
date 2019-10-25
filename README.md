# bert-multi-class
Evaluating BERT model for multi class classification

# requirements
 - pytorch
 - transformers (pip install transformers)
 - pandas
 - numpy
 - tqdm
 - jupyter
 - python3
 
# dataset format
In order to run this notebook training and tvaluation file should be in CSV format. Dataset files should contain "texts" and "labels" as two columns if you wanted to use the notebook out-of-box. Just change the path in the argument section.

# configuration
Default argument is mentioned in the notebook file but you can experiment with mentioned batch size. 

# caution
I have removed multi-gpu support code but code runs file on 1 GPU machine. I have tested this on AWS instances and Floydhub account.

Happy to answer questions if you have any.
