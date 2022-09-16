# PICUT EHR FINETUNING
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import pickle

from pathlib import Path
from tqdm.auto import tqdm
from tokenizers import Tokenizer, normalizers, processors
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from transformers import (
    AdamW, 
    BartConfig, 
    BartTokenizerFast,
    BartForConditionalGeneration, 
    BartForSequenceClassification, 
    get_linear_schedule_with_warmup
    )

import wandb
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

class EHRClassificationDataset(Dataset):
    def __init__(self, data_dir, column_class_name, tokenizer, max_seq_length, token_mask_prob, token_del_prob, seed=12345):
        self.seed = seed

        self.data_dir = data_dir
        self.data_files = [str(x) for x in Path(self.data_dir).glob('**/*.pkl')]
        self.max_seq_length = max_seq_length
        self.token_mask_prob = token_mask_prob
        self.token_del_prob = token_del_prob
        self.tokenizer = tokenizer
        self.column_class_name = column_class_name
        self.read_files()

    def __len__(self):
        return len(self.pid)

    def __getitem__(self, idx):
        pid = self.pid[idx]
        labels, labels_mask = self.tokenize(self.output_text[idx], max_length=self.max_seq_length)
        input_ids, attention_mask = self.tokenize(self.input_text[idx], max_length=self.max_seq_length)
        decoder_input_ids = self.create_decoder_input(input_ids=labels, pad_token_id=self.tokenizer.pad_token_id)
        class_name = self.class_name[idx]
        sample = {'pid': pid, 
                  'labels': labels, 
                  'labels_mask': labels_mask, 
                  'input_ids': input_ids, 
                  'attention_mask': attention_mask, 
                  'decoder_input_ids': decoder_input_ids, 
                  'class_name': class_name}
        return sample
    
    def read_files(self):
        data = []

        for path in self.data_files:
            with open(path, 'rb') as f:
                file_data = pickle.load(f)
            data.extend(file_data)
        
        self.pid = [pid["id"] for pid in data]
        self.original_text = [pid["original_text"] for pid in data]
        self.input_text = [pid["input_text"] for pid in data]
        self.output_text = [pid["output_text"] for pid in data]
        self.class_name = [pid[self.column_class_name] for pid in data]
    
    def tokenize(self, text, max_length=1024, padding='max_length', truncation=True):
        batch = self.tokenizer(text, max_length=max_length, padding=padding, truncation=truncation)

        input_ids = torch.tensor([x for x in batch["input_ids"]])
        mask = torch.tensor([x for x in batch["attention_mask"]])

        return (input_ids, mask)
    
    # This function is copied from modeling_bart.py
    def create_decoder_input(self, input_ids, pad_token_id):
        """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
        prev_output_tokens = input_ids.clone()
        index_of_eos = (input_ids.ne(pad_token_id).sum(dim=-1) - 1).unsqueeze(-1)
        prev_output_tokens[0] = input_ids.gather(-1, index_of_eos).squeeze()
        prev_output_tokens[1:] = input_ids[:-1]
        return prev_output_tokens

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('--class_weights_path', type=str, required=True, help="It should be a pickle file with a two float-element list.")
    parser.add_argument('--train_data_dir', type=str, required=True)
    parser.add_argument('--val_data_dir', type=str, required=True)
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--tokenizer_type', type=str, default="BPE", help="Choose between BytePair (BPE) and WorPiece (WP) algorithms")
    parser.add_argument(
        '--class_name', type=str, default="had_heart_failure", 
        help="The variable name for the classification task: had_heart_failure, hospitalized_three, hospitalized_seven, six_readmission, twelve_readmission"
        )
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--min_frequency', type=int, default=10)
    parser.add_argument('--shuffle_train', action='store_true', default=False)
    parser.add_argument('--token_mask_prob', type=float, default=0.15)
    parser.add_argument('--token_del_prob', type=float, default=0.01)
    parser.add_argument('--visit_del_prob', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--padding', type=str, default="max_lenght")
    parser.add_argument('--truncation', action='store_true', default=False)
    parser.add_argument('--add_prefix_space', action='store_true', default=False)
    parser.add_argument('--txt_dir', type=str, required=True)
    parser.add_argument('--tokenizer_dir', type=str, default="./tokenizer/")
    parser.add_argument('--untrained', action='store_true', default=False)

    # Model arguments
    parser.add_argument('--train_model', action='store_true', default=False)
    parser.add_argument('--start_from_previous_training', action='store_true', default=False)
    parser.add_argument('--activation_dropout', type=float, default=0.1)
    parser.add_argument('--activation_function', type=str, default="gelu")
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--classifier_dropout', type=float, default=0.1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--decoder_attention_heads', type=int, default=8)
    parser.add_argument('--decoder_ffn_dim', type=int, default=2048)
    parser.add_argument('--decoder_layerdrop', type=float, default=0.0)
    parser.add_argument('--decoder_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--encoder_attention_heads', type=int, default=8)
    parser.add_argument('--encoder_ffn_dim', type=int, default=2048)
    parser.add_argument('--encoder_layerdrop', type=float, default=0.0)
    parser.add_argument('--encoder_layers', type=int, default=3)
    parser.add_argument('--init_std', type=float, default=0.02)
    parser.add_argument('--is_encoder_decoder', action='store_true', default=False)
    parser.add_argument('--max_position_embeddings', type=int, default=512)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--scale_embedding', action='store_true', default=False)
    parser.add_argument('--use_cache', action='store_true', default=False)
    parser.add_argument('--bos_token_id', type=int, default=0)
    parser.add_argument('--pad_token_id', type=int, default=1)
    parser.add_argument('--eos_token_id', type=int, default=2)
    parser.add_argument('--decoder_start_token_id', type=int, default=2)
    parser.add_argument('--forced_eos_token_id', type=int, default=2)
    parser.add_argument('--attention_dropout', type=float, default=5e-5)
    
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--pretrained_model_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--class_head_epochs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--val_test_batch_size_ratio', type=int, default=2)
    parser.add_argument('--base_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warm_up_steps_ratio', type=float, default=0.2)

    # Other arguments
    parser.add_argument('--wandb_mode', type=str)
    parser.add_argument('--wandb_api_key', type=str)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--wandb_dir', type=str)

    return parser.parse_known_args()

def main():
    args, unknown = _parse_args()

    # set the special tokens for our transformer model
    args.unknown_token = '<unk>'
    args.special_tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>', '$', '&']

    with open(args.class_weights_path, 'rb') as f:
        args.class_weights = pickle.load(f)

    # Load data and create/load tokenizer from it
    if not os.path.exists(args.tokenizer_dir):
        # train dataset files
        paths = [str(x) for x in Path(args.txt_dir).glob('**/*.txt')]

        os.makedirs(args.tokenizer_dir)
        
        # initialize
        if args.tokenizer_type == "BPE":
            tokenizer = Tokenizer(BPE(unk_token = args.unknown_token))
        elif args.tokenizer_type == "WP":
            tokenizer = Tokenizer(WordPiece(unk_token = args.unknown_token))
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = Whitespace()

        # training
        if args.tokenizer_type == "BPE":
            trainer = BpeTrainer(vocab_size=args.vocab_size, min_frequency=args.min_frequency,
                            special_tokens=args.special_tokens)
        elif args.tokenizer_type == "WP":
            trainer = WordPieceTrainer(vocab_size=args.vocab_size, min_frequency=args.min_frequency,
                            special_tokens=args.special_tokens)

        tokenizer.train(paths, trainer) # training the tokenzier

        sos_token_id = tokenizer.token_to_id("<s>")
        eos_token_id = tokenizer.token_to_id("</s>")

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<s>:0 $A:0 </s>:0",
            special_tokens=[
                ("<s>", sos_token_id),
                ("</s>", eos_token_id),
            ],
        )

        tokenizer.save(args.tokenizer_dir + "/tokenizer-trained.json")

        tokenizer = BartTokenizerFast(tokenizer_object=tokenizer, max_length=args.max_length, padding=args.padding, \
        truncation=args.truncation, add_prefix_space=args.add_prefix_space)
    else:
        tokenizer = BartTokenizerFast(tokenizer_file=args.tokenizer_dir + "/tokenizer-trained.json", max_length=args.max_length, \
        padding=args.padding, truncation=args.truncation, add_prefix_space=args.add_prefix_space)

    args.vocab_size=tokenizer.vocab_size  # we align this to the tokenizer vocab_size

    # Training settings
    if args.untrained:
        args.pretrained_model_dir = None
        # Experiment name
        args.experiment_name = f"PICUT_{args.tokenizer_type}_finetune_{args.class_name}_{args.vocab_size}_vocab_non_pretrained"
    else:
        # Experiment name
        args.experiment_name = f"PICUT_{args.tokenizer_type}_finetune_{args.class_name}_{args.vocab_size}_vocab_non_pretrained"

    print(f"The arguments are: \n{args}")

    ## Data Loader
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load data
    trainDataset = EHRClassificationDataset(data_dir=args.train_data_dir, column_class_name=args.class_name, tokenizer=tokenizer, max_seq_length=args.max_length, 
                                token_mask_prob=args.token_mask_prob, token_del_prob=args.token_del_prob)
    validationDataset = EHRClassificationDataset(data_dir=args.val_data_dir, column_class_name=args.class_name, tokenizer=tokenizer, max_seq_length=args.max_length, 
                                token_mask_prob=args.token_mask_prob, token_del_prob=args.token_del_prob)
    testDataset = EHRClassificationDataset(data_dir=args.test_data_dir, column_class_name=args.class_name, tokenizer=tokenizer, max_seq_length=args.max_length, 
                                token_mask_prob=args.token_mask_prob, token_del_prob=args.token_del_prob)

    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle_train, drop_last=True)
    validationLoader = DataLoader(validationDataset, batch_size=args.batch_size // args.val_test_batch_size_ratio, num_workers=args.num_workers, drop_last=True)
    testLoader = DataLoader(testDataset, batch_size=args.batch_size // args.val_test_batch_size_ratio, num_workers=args.num_workers, drop_last=True)

    ## PICUT architecture model initialization
    configuration = BartConfig(
        activation_dropout=args.activation_dropout, 
        activation_function=args.activation_function, 
        attention_dropout=args.attention_dropout, 
        classifier_dropout=args.classifier_dropout, 
        d_model=args.d_model, 
        decoder_attention_heads=args.decoder_attention_heads, 
        decoder_ffn_dim=args.decoder_ffn_dim, 
        decoder_layerdrop=args.decoder_layerdrop, 
        decoder_layers=args.decoder_layers, 
        dropout=args.dropout, 
        encoder_attention_heads=args.encoder_attention_heads, 
        encoder_ffn_dim=args.encoder_ffn_dim, 
        encoder_layers=args.encoder_layers, 
        encoder_layerdrop=args.encoder_layerdrop, 
        init_std=args.init_std, 
        is_encoder_decoder=args.is_encoder_decoder, 
        max_position_embeddings=args.max_position_embeddings, 
        num_labels=args.num_labels, 
        scale_embedding=args.scale_embedding, 
        use_cache=args.use_cache, 
        vocab_size=args.vocab_size,  # we align this to the tokenizer vocab_size
        bos_token_id=args.bos_token_id, 
        pad_token_id=args.pad_token_id, 
        eos_token_id=args.eos_token_id, 
        decoder_start_token_id=args.decoder_start_token_id, 
        forced_eos_token_id=args.forced_eos_token_id,
    )

    # Create model with PICUT configurations
    pretrained_model = BartForConditionalGeneration(configuration)

    if not args.untrained:
        pretrained_model.load_state_dict(torch.load(args.pretrained_model_dir + 'picut_pretrain_model.pth', map_location="cuda:0"))

    # PICUT configurations adding Linear layer classifier
    model_class = BartForSequenceClassification(configuration)
    if not args.untrained:
        model_class.model = pretrained_model.model  ##not using pretrained weights

    # Accessing the model configuration
    configuration = model_class.config

    # Set the GPUs to be used
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_class = nn.DataParallel(model_class) 
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # and move our model over to the selected device
    model_class.to(device)

    optim = AdamW(
        model_class.parameters(), 
        lr=args.base_lr, 
        weight_decay=args.weight_decay)

    pp=0
    for p in list(model_class.parameters()):
        nnn=1
        for s in list(p.size()):
            nnn = nnn*s
        pp += nnn
    print(f"Total model parameters: {pp}")

    # Total number of training steps
    epoch_train_steps = len(trainLoader)
    epoch_val_steps = len(validationLoader)
    epoch_test_steps = len(testLoader)
    args.total_steps = epoch_train_steps * args.epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=int(args.total_steps * args.warm_up_steps_ratio),
                                                num_training_steps=args.total_steps)

    # Loss function
    weights = torch.tensor(args.class_weights).float().to(device)
    criterion_weighted = nn.CrossEntropyLoss(weight=weights)

    ## Training
    # Start a new run
    wandb.init(name=args.experiment_name)

    # Save model inputs and hyperparameters
    config = wandb.config
    config.update(args)

    model_dir = args.model_dir

    min_class_weight = np.min(args.class_weights)
    stop_training = 0
    min_valid_loss = np.inf
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    # Log gradients and model parameters
    wandb.watch(model_class)

    for epoch in range(args.epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(trainLoader, leave=True)
        val_loop = tqdm(validationLoader, leave=True)

        train_loss = 0.0
        train_acc = 0.0
        train_batch = 1 + (epoch * epoch_train_steps)
        train_labels = np.array([])
        train_pred_probs = np.array([])

        # activate training mode
        model_class.train()

        if epoch <= (args.class_head_epochs - 1):
            for param in model_class.model.parameters():
                param.requires_grad = False
        else:
            for param in model_class.model.parameters():
                param.requires_grad = True

        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch[args.class_name].to(device)
            
            # process
            outputs = model_class(input_ids, attention_mask=attention_mask)

            # loss = outputs.loss
            loss = criterion_weighted(outputs.logits.float(), labels)

            acc = torch.sum(labels == torch.argmax(outputs.logits, dim=-1), dim=-1) / labels.shape[0]
            train_lr = optim.param_groups[0]["lr"]
            
            prob_layer = nn.Softmax(dim=1)
            pred_prob = prob_layer(outputs.logits).cpu().detach().numpy()
            train_pred_probs = np.concatenate([train_pred_probs, pred_prob[:,1]])
            train_labels = np.concatenate([train_labels, labels.cpu().detach().numpy()])

            wandb.log({
                "train_batch_step": train_batch, 
                "loss": torch.mean(loss).item(), 
                "accuracy": torch.mean(acc).item(), 
                "learning_rate": train_lr
                })

            # calculate loss for every parameter that needs grad update
            loss.sum().backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model_class.parameters(), 1.0)

            # update parameters
            optim.step()
            scheduler.step()
            
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch+1}')
            loop.set_postfix(loss=torch.mean(loss).item(), acc=torch.mean(acc).item(), lr=train_lr)

            # Calculate Loss
            train_loss += torch.mean(loss).item()
            train_acc += torch.mean(acc).item()
            train_batch += 1

        valid_loss = 0.0
        valid_acc = 0.0
        valid_batch = 1 + (epoch * epoch_val_steps)
        valid_labels = np.array([])
        valid_pred_probs = np.array([])
        
        # activate evaluation mode
        model_class.eval()

        for batch in val_loop:
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch[args.class_name].to(device)
            
            # infer
            with torch.no_grad():
                outputs = model_class(input_ids, attention_mask=attention_mask)
            
            val_loss = criterion_weighted(outputs.logits.float(), labels)

            val_acc = torch.sum(labels == torch.argmax(outputs.logits, dim=-1), dim=-1) / labels.shape[0]
            
            prob_layer = nn.Softmax(dim=1)
            pred_prob = prob_layer(outputs.logits).cpu().detach().numpy()
            valid_pred_probs = np.concatenate([valid_pred_probs, pred_prob[:,1]])
            valid_labels = np.concatenate([valid_labels, labels.cpu().detach().numpy()])
            
            wandb.log({
                "val_batch_step": valid_batch, 
                "val_loss": torch.mean(val_loss).item(), 
                "val_accuracy": torch.mean(val_acc).item()
                })

            val_loop.set_description(f'Epoch {epoch+1}')
            val_loop.set_postfix(val_loss=torch.mean(val_loss).item(), val_acc=torch.mean(val_acc).item())
            valid_loss += torch.mean(val_loss).item()
            valid_acc += torch.mean(val_acc).item()
            valid_batch += 1
        
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / epoch_train_steps} \t\t Training Accuracy: {train_acc / epoch_train_steps} \n\
            \t\t Validation Loss: {valid_loss / epoch_val_steps} \t\t Validation Accuracy: {valid_acc / epoch_val_steps}')
        train_losses.append(train_loss / epoch_train_steps)
        train_accuracies.append(train_acc / epoch_train_steps)
        valid_losses.append(valid_loss / epoch_val_steps)
        valid_accuracies.append(valid_acc / epoch_val_steps)

        train_pred_labels = np.array([1 if p > 0.5 else 0 for p in train_pred_probs])
        valid_pred_labels = np.array([1 if p > 0.5 else 0 for p in valid_pred_probs])
        train_sample_weight = [float(args.class_weights[int(cl)] / min_class_weight) for cl in train_labels]
        valid_sample_weight = [float(args.class_weights[int(cl)] / min_class_weight) for cl in valid_labels]

        train_auc_score = roc_auc_score(train_labels, train_pred_probs)
        train_aps_score = average_precision_score(train_labels, train_pred_probs)
        train_f1_score = f1_score(train_labels, train_pred_labels, sample_weight=train_sample_weight, average='binary')
        valid_auc_score = roc_auc_score(valid_labels, valid_pred_probs)
        valid_aps_score = average_precision_score(valid_labels, valid_pred_probs)
        valid_f1_score = f1_score(valid_labels, valid_pred_labels, sample_weight=valid_sample_weight, average='binary')

        wandb.log({
            "epoch": epoch+1, 
            "epoch_loss": train_loss / epoch_train_steps, 
            "epoch_accuracy": train_acc / epoch_train_steps, 
            "epoch_auc_score": train_auc_score, 
            "epoch_f1_score": train_f1_score, 
            "epoch_aps_score": train_aps_score, 
            "epoch_val_loss": valid_loss / epoch_val_steps, 
            "epoch_val_accuracy": valid_acc / epoch_val_steps, 
            "epoch_val_auc_score": valid_auc_score, 
            "epoch_val_f1_score": valid_f1_score, 
            "epoch_val_aps_score": valid_aps_score
            })

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            
            # Saving State Dict
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            torch.save(model_class.state_dict(), model_dir + 'picut_finetune_model.pth')
            stop_training = 0
        else:
            stop_training += 1
        
        if stop_training == 3:
            print("Training stopped after three epochs without better validation loss.")
            break

    # Load best model
    model_class = BartForSequenceClassification(configuration)
    model_class.load_state_dict(torch.load(model_dir + 'picut_finetune_model.pth', map_location="cuda:0"))

    # and move our model over to the selected device
    model_class.to(device)

    ## Testing
    test_loop = tqdm(testLoader, leave=True)
    testing_loss = 0.0
    testing_acc = 0.0
    test_batch = 1
    test_labels = np.array([])
    test_pred_probs = np.array([])

    for batch in test_loop:
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch[args.class_name].to(device)

        # infer
        with torch.no_grad():
            outputs = model_class(input_ids, attention_mask=attention_mask)

        test_loss = criterion_weighted(outputs.logits.float(), labels)

        test_acc = torch.sum(labels == torch.argmax(outputs.logits, dim=-1), dim=-1) / labels.shape[0]
        
        prob_layer = nn.Softmax(dim=1)
        pred_prob = prob_layer(outputs.logits).cpu().detach().numpy()
        test_pred_probs = np.concatenate([test_pred_probs, pred_prob[:,1]])
        test_labels = np.concatenate([test_labels, labels.cpu().detach().numpy()])

        wandb.log({
            "test_batch_step": test_batch, 
            "test_loss": torch.mean(test_loss).item(), 
            "test_accuracy": torch.mean(test_acc).item()
            })

        test_loop.set_description(f'Testing: ')
        test_loop.set_postfix(test_loss=torch.mean(test_loss).item(), test_acc=torch.mean(test_acc).item())
        
        testing_loss += torch.mean(test_loss).item()
        testing_acc += torch.mean(test_acc).item()
        test_batch += 1

    test_pred_labels = np.array([1 if p > 0.5 else 0 for p in test_pred_probs])
    test_sample_weight = [float(args.class_weights[int(cl)] / min_class_weight) for cl in test_labels]

    test_auc_score = roc_auc_score(test_labels, test_pred_probs)
    test_aps_score = average_precision_score(test_labels, test_pred_probs)
    test_f1_score = f1_score(test_labels, test_pred_labels, sample_weight=test_sample_weight, average='binary')

    print(f'Testing Loss: {testing_loss / epoch_test_steps} \t\t Testing Accuracy: {testing_acc / epoch_test_steps}')
    wandb.log({
        "test_mean_loss": testing_loss / epoch_test_steps, 
        "test_mean_accuracy": testing_acc / epoch_test_steps, 
        "test_auc_score": test_auc_score, 
        "test_f1_score": test_f1_score, 
        "test_aps_score": test_aps_score
        })

    wandb.finish()


if __name__ == "__main__":

    main()