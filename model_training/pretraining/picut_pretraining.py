# PICUT EHR PRETRAINING
import sys
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import argparse

import wandb
from pathlib import Path
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
    get_linear_schedule_with_warmup
)

print("GPU used: ", os.getenv("CUDA_VISIBLE_DEVICES"))

class EHRDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_seq_length, token_mask_prob, token_del_prob, visit_del_prob, seed=12345):
        self.seed = seed

        self.data_dir = data_dir
        self.data_files = [str(x) for x in Path(self.data_dir).glob('**/*.pkl')]
        self.max_seq_length = max_seq_length
        self.token_mask_prob = token_mask_prob
        self.token_del_prob = token_del_prob
        self.visit_del_prob = visit_del_prob
        self.tokenizer = tokenizer
        self.read_files()

    def __len__(self):
        return len(self.pid)

    def __getitem__(self, idx):
        pid = self.pid[idx]
        input_text = self.input_text[idx]
        input_text = self.random_visit_deletion(input_text, self.visit_del_prob)
        input_text = self.visit_shuffling(input_text)
        input_text = self.shift_text_right(input_text)
        
        labels, labels_mask = self.tokenize(self.output_text[idx], max_length=self.max_seq_length)
        input_ids, attention_mask = self.tokenize(input_text, max_length=self.max_seq_length)
        input_ids = self.mask_tokens(input_ids=input_ids, mask_prob=self.token_mask_prob, mask_token_id=self.tokenizer.mask_token_id)
        input_ids, attention_mask = self.text_infilling_mask_tokens(input_ids=input_ids)
        input_ids, attention_mask = self.delete_input_tokens(input_ids=input_ids, attention_mask=attention_mask, del_prob=self.token_del_prob)
        decoder_input_ids = self.create_decoder_input(input_ids=labels, pad_token_id=self.tokenizer.pad_token_id)

        sample = {'pid': pid, 
                  'labels': labels, 
                  'labels_mask': labels_mask, 
                  'input_ids': input_ids, 
                  'attention_mask': attention_mask, 
                  'decoder_input_ids': decoder_input_ids
                  }
        return sample
    
    def read_files(self):
        data = []

        for path in self.data_files:
            with open(path, 'rb') as f:
                file_data = pickle.load(f)
            data.extend(file_data)
        
        self.pid = [pid["id"] for pid in data]
        self.demographic_info = [pid["demographic_info"] for pid in data]
        self.original_text = [pid["original_text"] for pid in data]
        self.input_text = [pid["input_text"] for pid in data]
        self.output_text = [pid["output_text"] for pid in data]
    
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
    
    def mask_tokens(self, input_ids, mask_prob=0.15, mask_token_id=4):
        """Mask a percentage of tokens (usually <mask>)."""
        # torch.manual_seed(self.seed)
        prev_output_tokens = input_ids.clone()

        if self.token_mask_prob is not None:
            mask_prob = self.token_mask_prob

        # create random array of floats with equal dims to input_ids
        input_to_be_masked = torch.ones_like(input_ids) * input_ids.ne(self.tokenizer.bos_token_id) * input_ids.ne(self.tokenizer.eos_token_id) * \
            input_ids.ne(self.tokenizer.pad_token_id) * input_ids.ne(self.tokenizer.mask_token_id) * \
                input_ids.ne(self.tokenizer.unk_token_id) * input_ids.ne(self.tokenizer.vocab["$"]) * input_ids.ne(self.tokenizer.vocab["&"])
        rand = torch.rand(input_ids.shape) * input_to_be_masked

        # mask random 15% where token is not special tokens
        mask_arr = (rand < mask_prob) * (rand != 0)
        
        # get indices of mask positions from mask array
        selection = torch.flatten(mask_arr.nonzero()).tolist()
        # mask input_ids
        prev_output_tokens[selection] = mask_token_id  # our custom [MASK] token == 4
        return prev_output_tokens
    
    def text_infilling_mask_tokens(self, input_ids):
        """Mask a series of tokens (usually <mask>)."""
        # torch.manual_seed(self.seed)
        prev_output_tokens = input_ids.clone()

        unique_input_ids = torch.unique_consecutive(prev_output_tokens)

        unique_input_ids = torch.cat((unique_input_ids, torch.ones(len(prev_output_tokens) - len(unique_input_ids), dtype=torch.int)))

        unique_attention_mask = unique_input_ids.ne(1) * torch.ones_like(prev_output_tokens)

        return unique_input_ids, unique_attention_mask

    def shift_text_right(self, text):
        """Shift input ids some text strings to the right."""
        prev_text = text.split(" ")
        demo_index = prev_text.index("$")
        num_rot = int(torch.randint(low=0, high=len(prev_text) - demo_index, size=(1,)))

        rot_text = prev_text[demo_index + 1:]
        prev_text = prev_text[:demo_index + 1]
        for _ in range(num_rot):
            temp = rot_text[1:]
            temp.append(rot_text[0])
            rot_text = temp
        new_text = prev_text + rot_text
        return " ".join(new_text)

    def visit_shuffling(self, text):
        """Shuffle input clinical history randomly."""
        prev_text = text.split(" ")
        demo_index = prev_text.index("$")
        shu_text = prev_text[demo_index + 1:]
        prev_text = prev_text[:demo_index + 1]
        shu_text = " ".join(shu_text)
        shu_text = shu_text.split(" & ")
        shu_text = random.sample(shu_text, len(shu_text))
        shu_text = " & ".join(shu_text)
        shu_text = shu_text.split(" ")

        new_text = prev_text + shu_text
        return " ".join(new_text)

    def random_visit_deletion(self, text, visit_del_prob):
        """Delete visits randomly from input clinical history."""
        prev_text = text.split(" ")
        demo_index = prev_text.index("$")
        visit_text = prev_text[demo_index + 1:]
        prev_text = prev_text[:demo_index + 1]
        visit_text = " ".join(visit_text)
        visit_text = visit_text.split(" & ")
        no_visits = len(visit_text)
        if no_visits > 1:
            to_delete = [random.random() for _ in range(no_visits)]
            visit_text = [x for i, x in enumerate(visit_text) if not to_delete[i] < visit_del_prob]
        visit_text = " & ".join(visit_text)
        visit_text = visit_text.split(" ")

        new_text = prev_text + visit_text
        return " ".join(new_text)
    
    def delete_input_tokens(self, input_ids, attention_mask, del_prob = 0.1):
        prev_output_tokens = input_ids.clone()
        prev_attention_mask = attention_mask.clone()

        if self.token_del_prob is not None:
            del_prob = self.token_del_prob

        # randomly delete some tokens
        rand = torch.rand(input_ids.shape) * input_ids.ne(self.tokenizer.bos_token_id) * input_ids.ne(self.tokenizer.eos_token_id) * \
            input_ids.ne(self.tokenizer.pad_token_id) * input_ids.ne(self.tokenizer.mask_token_id) * \
                input_ids.ne(self.tokenizer.unk_token_id) * input_ids.ne(self.tokenizer.vocab["$"]) * input_ids.ne(self.tokenizer.vocab["&"])
        
        del_mask = (rand < del_prob) * (rand != 0)

        mask = torch.ones(input_ids.numel(), dtype=torch.bool)
        mask[del_mask] = False

        del_input_ids = prev_output_tokens[mask]
        del_attention_mask = prev_attention_mask[mask]

        del_input_ids = torch.cat((del_input_ids, torch.ones(len(prev_output_tokens) - len(del_input_ids), dtype=torch.int)))
        del_attention_mask = torch.cat((del_attention_mask, torch.zeros(len(prev_attention_mask) - len(del_attention_mask), dtype=torch.int)))
        
        return del_input_ids, del_attention_mask   

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('--train_data_dir', type=str, required=True)
    parser.add_argument('--val_data_dir', type=str, required=True)
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--tokenizer_type', type=str, default="BPE", help="Choose between BytePair (BPE) and WorPiece (WP) algorithms")
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
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--val_test_batch_size_ratio', type=int, default=2)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warm_up_steps_ratio', type=float, default=0.1)

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

    # Experiment name
    args.experiment_name = f"PICUT_{args.tokenizer_type}_{args.vocab_size}_vocab"

    print(f"The arguments are: \n{args}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create dataset and dataloader objects for later training
    trainDataset = EHRDataset(data_dir=args.train_data_dir, tokenizer=tokenizer, max_seq_length=args.max_length, 
                                token_mask_prob=args.token_mask_prob, token_del_prob=args.token_del_prob, visit_del_prob=args.visit_del_prob)
    validationDataset = EHRDataset(data_dir=args.val_data_dir, tokenizer=tokenizer, max_seq_length=args.max_length, 
                                token_mask_prob=args.token_mask_prob, token_del_prob=args.token_del_prob, visit_del_prob=args.visit_del_prob)
    testDataset = EHRDataset(data_dir=args.test_data_dir, tokenizer=tokenizer, max_seq_length=args.max_length, 
                                token_mask_prob=args.token_mask_prob, token_del_prob=args.token_del_prob, visit_del_prob=args.visit_del_prob)

    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle_train, drop_last=True)
    validationLoader = DataLoader(validationDataset, batch_size=args.batch_size // args.val_test_batch_size_ratio, num_workers=args.num_workers, drop_last=True)
    testLoader = DataLoader(testDataset, batch_size=args.batch_size // args.val_test_batch_size_ratio, num_workers=args.num_workers, drop_last=True)

    ## PICUT architecture initialization model
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

    # Create model with input configurations
    model = BartForConditionalGeneration(configuration)

    # Accessing the model configuration
    configuration = model.config

    # Set the GPUs to be used
    if torch.cuda.device_count() > 1:
        cuda_count = torch.cuda.device_count()
        print("Let's use", cuda_count, "GPUs!")
        model = nn.DataParallel(model, device_ids=[i for i in range(cuda_count)])
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # and move our model over to the selected device
    model.to(device)

    # Load previous model if needed
    if args.start_from_previous_training:
        model.load_state_dict(torch.load(args.model_dir + 'picut_pretrain_model.pth'))

    optim = AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    pp=0
    for p in list(model.parameters()):
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

    # Label Smoothing cross entropy loss
    def linear_combination(x, y, epsilon): 
        return epsilon*x + (1-epsilon)*y

    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

    #  Implementation of Label smoothing with CrossEntropy and ignore_index
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, epsilon:float=0.1, reduction='mean',ignore_index=-100):
            super().__init__()
            self.epsilon = epsilon
            self.reduction = reduction
            self.ignore_index = ignore_index
        def forward(self, preds, target):
            n = preds.size()[-1]
            log_preds = torch.nn.functional.log_softmax(preds, dim=-1)
            loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
            nll = torch.nn.functional.nll_loss(log_preds, target, reduction=self.reduction,ignore_index=self.ignore_index, )
            return linear_combination(loss/n, nll, self.epsilon)

    ## Choose loss function
    criterion = LabelSmoothingCrossEntropy(ignore_index=tokenizer.pad_token_id)

    ## Training
    wandb.init(name=args.experiment_name)

    # Save model inputs and hyperparameters
    config = wandb.config
    config.update(args)

    stop_training = 0
    min_valid_loss = np.inf
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    # Log gradients and model parameters
    wandb.watch(model)

    if args.train_model:
        for epoch in range(args.epochs):
            # setup loop with TQDM and dataloader
            loop = tqdm(trainLoader, leave=True)
            val_loop = tqdm(validationLoader, leave=True)

            train_loss = 0.0
            train_acc = 0.0
            train_batch = 1 + (epoch * epoch_train_steps)

            # activate training mode
            model.train()

            for batch in loop:
                # initialize calculated gradients (from prev step)
                optim.zero_grad()
                
                # pull all tensor batches required for training
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                labels_mask = batch['labels_mask'].to(device)
                
                # process
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                ## Using in-built loss function from Hugging Face
                # loss = outputs.loss

                ## Using define loss function
                loss = criterion(outputs.logits.view(-1, configuration.vocab_size), labels.view(-1))

                acc = torch.sum(labels[labels_mask.ne(0)] == torch.argmax(outputs.logits, dim=-1)[labels_mask.ne(0)], dim=-1) /torch.sum(labels_mask.ne(0))
                train_lr = optim.param_groups[0]["lr"]

                wandb.log({"train_batch_step": train_batch, "loss": torch.mean(loss).item(), "accuracy": torch.mean(acc).item(), "learning_rate": train_lr})

                # calculate loss for every parameter that needs grad update
                loss.sum().backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # update parameters
                optim.step()
                scheduler.step()

                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch+1}')
                # loop.set_postfix(loss=loss.item(), acc=acc.item())
                loop.set_postfix(loss=torch.mean(loss).item(), acc=torch.mean(acc).item(), lr=train_lr)

                # Calculate Loss
                # train_loss += loss.item()
                train_loss += torch.mean(loss).item()
                train_acc += torch.mean(acc).item()
                train_batch += 1

            valid_loss = 0.0
            valid_acc = 0.0
            valid_batch = 1 + (epoch * epoch_val_steps)
            
            # activate evaluation mode
            model.eval()

            for batch in val_loop:
                # pull all tensor batches required for training
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                labels_mask = batch['labels_mask'].to(device)
                
                # infer
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                ## Using in-built loss function from Hugging Face
                # val_loss = outputs.loss

                ## Using define loss function
                val_loss = criterion(outputs.logits.view(-1, configuration.vocab_size), labels.view(-1))
                
                val_acc = torch.sum(labels[labels_mask.ne(0)] == torch.argmax(outputs.logits, dim=-1)[labels_mask.ne(0)], dim=-1) /torch.sum(labels_mask.ne(0))
                
                wandb.log({"val_batch_step": valid_batch, "val_loss": torch.mean(val_loss).item(), "val_accuracy": torch.mean(val_acc).item()})

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

            wandb.log({"epoch": epoch+1, "epoch_loss": train_loss / epoch_train_steps, "epoch_accuracy": train_acc / epoch_train_steps, 
                "epoch_val_loss": valid_loss / epoch_val_steps, "epoch_val_accuracy": valid_acc / epoch_val_steps})

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                
                # Saving State Dict
                if not os.path.exists(args.model_dir):
                    os.makedirs(args.model_dir)

                torch.save(model.state_dict(), args.model_dir + 'picut_pretrain_model.pth')
                stop_training = 0
            else:
                stop_training += 1
            
            if stop_training == 3:
                print("Training stopped after three epochs without better validation loss.")
                break

    ## Load best trained model
    model.load_state_dict(torch.load(args.model_dir + 'picut_pretrain_model.pth', map_location="cuda:0"))
    model.to(device)

    ## Testing
    test_loop = tqdm(testLoader, leave=True)
    testing_loss = 0.0
    testing_acc = 0.0
    test_batch = 1

    # activate evaluation mode
    model.eval()

    for batch in test_loop:
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        labels_mask = batch['labels_mask'].to(device)
        
        # infer
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        ## Using in-built loss function from Hugging Face
        # test_loss = outputs.loss

        ## Using define loss function
        test_loss = criterion(outputs.logits.view(-1, configuration.vocab_size), labels.view(-1))
        
        test_acc = torch.sum(labels[labels_mask.ne(0)] == torch.argmax(outputs.logits, dim=-1)[labels_mask.ne(0)], dim=-1) /torch.sum(labels_mask.ne(0))
        
        wandb.log({"test_batch_step": test_batch, "test_loss": torch.mean(test_loss).item(), "test_accuracy": torch.mean(test_acc).item()})

        test_loop.set_description(f'Evaluating')
        test_loop.set_postfix(test_loss=torch.mean(test_loss).item(), test_acc=torch.mean(test_acc).item())
        
        testing_loss += torch.mean(test_loss).item()
        testing_acc += torch.mean(test_acc).item()
        test_batch += 1

    print(f'Testing Loss: {testing_loss / epoch_test_steps} \t\t Testing Accuracy: {testing_acc / epoch_test_steps}')
    wandb.log({"test_mean_loss": testing_loss / epoch_test_steps, "test_mean_accuracy": testing_acc / epoch_test_steps})

    wandb.finish()


if __name__ == "__main__":

    main()