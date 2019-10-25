import pandas as pd
import uuid
import os
import random
from argparse import Namespace

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer)
from transformers import AdamW, WarmupLinearSchedule

MODEL_CLASSES = { 'bert': (BertConfig, BertForSequenceClassification, BertTokenizer) }
import logging

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid   = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label  = label

class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label          = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
        
    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError() 

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class MultiClassProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def __init__(self, train_filepath, dev_filepath, test_filepath):
        self.train_filepath = train_filepath
        self.dev_filepath   = dev_filepath
        self.test_filepath  = test_filepath

    def get_train_examples(self):
        """See base class."""
        df            = self._get_dataframe(self.train_filepath)
        return self._get_examples(df)

    def get_dev_examples(self):
        """See base class."""
        df            = self._get_dataframe(self.dev_filepath)
        return self._get_examples(df)
    
    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        df            = self._get_dataframe(self.test_filepath)
        return self._get_examples(df)

    def get_labels(self):
        """See base class."""
        df            = pd.read_csv(self.train_filepath)
        self.labels   = list(df.labels.unique())
        return self.labels
    
    def _get_dataframe(self, filepath):
        df            = pd.read_csv(filepath)
        return df

    def _get_examples(self, df):
        examples = []
        for index, row in df.iterrows():
            examples.append(InputExample(guid=str(uuid.uuid4()), text_a=row['texts'], text_b=None, label=row['labels']))
        return examples


def convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None, 
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        label = label_map[example.label]

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))
    return features

def get_examples_dataset(examples, labels, m_tokenzier, max_seq_length):
    features = convert_examples_to_features(examples,
                                            m_tokenzier,
                                            label_list=labels,
                                            max_length=max_seq_length,
                                            pad_on_left=False,
                                            pad_token=m_tokenzier.convert_tokens_to_ids([m_tokenzier.pad_token])[0],
                                            pad_token_segment_id=0,
    )


    # Convert to Tensors and build dataset
    all_input_ids       = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask  = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids  = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels          = torch.tensor([f.label for f in features], dtype=torch.long)
    
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size   = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler           = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader        = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total                 = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                
                if global_step % 100 == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    
    # save 
    return global_step, tr_loss / global_step

def evaluate(args, eval_dataset, model, tokenizer):
    results = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    result = compute_metrics("eval_task", preds, out_label_ids)
    results.update(result)

    return results

def main():
    args = Namespace (
            n_gpu=1,
            seed=1337,
            train_batch_size=8,
            per_gpu_train_batch_size=8,
            per_gpu_eval_batch_size=8,
            local_rank=-1,
            max_seq_length=128,
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            weight_decay=0.0,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            num_train_epochs=1.0,
            max_steps=-1,
            warmup_steps=0,
            model_type='bert',
            data_dir='/Users/kd/Workspace/python/data/tt',
            output_dir='/Users/kd/Downloads/bert-model',
            train_filepath='',
            valid_filepath='',
            test_filepath='',
            config_name='bert-base-uncased',
            tokenizer_name='bert-base-uncased',
            do_lower_case=True,
            cuda=True,
            do_eval=False,
            do_train=True
        )

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("Using CUDA: {}".format(args.cuda))

    args.train_filepath = os.path.join(args.data_dir, 'train.csv')
    args.valid_filepath = os.path.join(args.data_dir, 'valid.csv')
    args.test_filepath  = os.path.join(args.data_dir, 'test.csv')

    ### Training start ###
    if args.do_train:
        set_seed(args)
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

        processor       = MultiClassProcessor(args.train_filepath, args.valid_filepath, args.test_filepath)
        label_list      = processor.get_labels()
        train_examples  = processor.get_train_examples()

        config          = config_class.from_pretrained(args.config_name, num_labels=len(label_list))
        tokenizer       = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)
        model           = model_class.from_pretrained(args.config_name, config=config).to(args.device)

        print(tokenizer, tokenizer_class)
        train_dataset   = get_examples_dataset(train_examples, label_list, tokenizer, args.max_seq_length)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    #### Training end ####
    if args.do_eval:
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

        processor       = MultiClassProcessor(args.train_filepath, args.valid_filepath, args.test_filepath)
        label_list      = processor.get_labels()
        eval_examples   = processor.get_dev_examples()

        checkpoint      = '/Users/kd/Downloads/bert-model/checkpoint-400'
        tokenizer       = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model           = model_class.from_pretrained(checkpoint).to(args.device)

        eval_dataset    = get_examples_dataset(eval_examples, label_list, tokenizer, args.max_seq_length)
        result          = evaluate(args, eval_dataset, model, tokenizer)
        logger.info(" evaluation result = %s", result)

if __name__ == "__main__":
    main()