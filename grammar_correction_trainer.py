from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import  DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torch.optim import AdamW
import torch
from utils import  set_trainable_layers
import numpy as np
from tqdm import tqdm
import copy
from config import settings
from torch.utils.data import  DataLoader
import optuna
import copy 



class GrammarCorrectionTrainer:

    def __init__(
            self, 
            train_loader : DataLoader, 
            es_loader : DataLoader, 
            valid_loader: DataLoader, 
            test_loader : DataLoader,
            full_train_loader: DataLoader
    ):
        """
        Grammar Correction Trainer class for training and evaluating a grammar correction model.

        Args:
            train_path (str): Path to the training dataset CSV file.
            es_path (str): Path to the early stopping dataset CSV file.
            valid_path (str): Path to the validation dataset CSV file.
            test_path (str): Path to the test dataset CSV file.
        """
        # Set hyperparameters and device
        self.train_loader = train_loader
        self.es_loader = es_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.full_train_loader = full_train_loader

        self.epoch_num = settings.epoch_num
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokeniser = AutoTokenizer.from_pretrained(settings.base_model)
        
    def create_batch(self,X, y):
        """
        Creates a batch of tokenized input and target sequences for model training.
        Args:
            X (list of str): List of input sequences to be corrected.
            y (list of str): List of target (corrected) sequences.
        Returns:
            dict: A dictionary containing tokenized input tensors and labels suitable for model training.
                - The input tensors are tokenized and moved to the appropriate device.
                - The labels tensor replaces padding token IDs with -100 for loss masking.
        """
        # Tokenize the input sequences with task prefix and target sequences
        batch = self.tokeniser(["correct grammar: " + sequence for sequence in X], padding=True, truncation=True, return_tensors="pt").to(self.device)
        # Tokenize the target sequences and mask padding tokens
        labels = self.tokeniser(y, padding=True, truncation=True, return_tensors="pt").to(self.device).input_ids
        labels[labels== self.tokeniser.pad_token_id] = -100  # Mask padding tokens for loss calculation
        batch['labels'] = labels
        return batch

    def epoch_train(self, model, train_loader, optimiser, scheduler):
        
        # Initialize lists to store training losses and predictions
        train_losses = []
        train_references = []
        train_hypotheses = []

        # Training phase of the epoch
        with tqdm(range(len(train_loader))) as ptrain_bar:
            for X_train, y_train in train_loader:
                model.train()
                train_batch = self.create_batch(X_train, y_train)
                # Passing training batch to the model to get loss
                train_loss = model(**train_batch, output_attentions=False, output_hidden_states=False).loss

                # Backpropagation and optimization step
                optimiser.zero_grad()
                train_loss.backward()
                optimiser.step()
                scheduler.step()
                
                # Generate predictions for BLEU calculation and decode them for BLEU calculation
                model.eval()
                train_output = model.generate(input_ids = train_batch['input_ids'], attention_mask = train_batch["attention_mask"])
                hypotheses = self.tokeniser.batch_decode(train_output, skip_special_tokens=True)
                train_references += [[sentence.split()] for sentence in y_train]
                train_hypotheses += [hypothesis.split() for hypothesis in hypotheses]

                train_losses.append(train_loss.detach().cpu().item())
                ptrain_bar.update(1)

            

        return train_losses, train_references, train_hypotheses
    
    def epoch_eval(self, model, data_loader):

        # Initialize lists to store evaluation losses and predictions
        model.eval()
        eval_references = []   
        eval_hypotheses = []
        eval_losses = []
        
        with torch.no_grad(), tqdm(range(len(data_loader))) as peval_bar:
            # Iterate through the evaluation data loader
            for X, y, in data_loader:
                eval_batch = self.create_batch(X, y)
                # Passing early stopping batch to the model to get loss
                eval_loss = model(**eval_batch, output_attentions=False, output_hidden_states=False).loss
                
                # Decoding predictions for BLEU calculation
                eval_output = model.generate(input_ids = eval_batch["input_ids"], attention_mask = eval_batch["attention_mask"])
                hypotheses = self.tokeniser.batch_decode(eval_output, skip_special_tokens=True)
                eval_references +=  [[sentence.split()] for sentence in y]
                eval_hypotheses += [hypothesis.split() for hypothesis in hypotheses]
            
                eval_losses.append(eval_loss.cpu().item())
                peval_bar.update(1)

        return eval_losses, eval_references, eval_hypotheses


    def hyperparameter_tuning(self):
        def objective(trial):
            # Suggest hyperparameters for the model
            n_decoder_unfreeze = trial.suggest_categorical("n_decoder_unfreeze", settings.hyperparams.decoder_unfreeze)
            n_encoder_unfreeze = trial.suggest_categorical("n_encoder_unfreeze", settings.hyperparams.encoder_unfreeze)
            learning_rate = trial.suggest_categorical("learning_rate", settings.hyperparams.learning_rate)
            
            # Load the model and set trainable layers
            model = AutoModelForSeq2SeqLM.from_pretrained(settings.base_model)
            model.to(self.device)
            set_trainable_layers(model, n_decoder_unfreeze, n_encoder_unfreeze)

            # Optimizer and scheduler setup
            optimiser = AdamW(filter(lambda x : x.requires_grad, model.parameters()), lr=learning_rate)
            total_steps = self.epoch_num * len(self.train_loader)
            scheduler = get_linear_schedule_with_warmup(optimiser, num_warmup_steps=0, num_training_steps=total_steps)

            # Stats tracking
            train_stats = {"train_loss" : [], "train_bleu": [], "es_loss" : [], "es_bleu" : []}
            best_epoch_idx = None
            min_val_loss = float('inf')
            epoch_no_update = 0

            # Iterate though each training epoch
            for epoch_idx in range(self.epoch_num):
                # Train the model on the training data
                train_losses, train_references, train_hypotheses = self.epoch_train(model, self.train_loader, optimiser, scheduler)
                # Evaluate the model on the early stopping data
                es_losses , es_references, es_hypotheses = self.epoch_eval(model, self.es_loader)

                # Calculate BLEU scores and update stats
                es_bleu = corpus_bleu(es_references, es_hypotheses)
                train_bleu = corpus_bleu(train_references, train_hypotheses)
                train_stats["train_bleu"].append(train_bleu)
                train_stats["es_bleu"].append(es_bleu)

                epoch_stats = {'train_loss' : train_losses, "es_loss" : es_losses}
                # Store mean train and early stopping losses for the epoch
                for key, value in epoch_stats.items():
                    mean_val = np.mean(value)
                    train_stats[key].append(mean_val)

                epoch_train_loss = train_stats["train_loss"][-1]
                epoch_es_loss = train_stats["es_loss"][-1]

                print("Epoch {:d} - train_loss: {:.4f}, train_bleu : {:.4f}, es_loss: {:.4f}, es_bleu: {:.4f}".format(epoch_idx,epoch_train_loss, train_bleu, epoch_es_loss, es_bleu))

                # Early stopping logic to save the best model
                if epoch_es_loss < min_val_loss:
                    min_val_loss = epoch_es_loss
                    best_model_param = copy.deepcopy(model.state_dict())
                    best_epoch_idx = epoch_idx
                    epoch_no_update = 0 
                else:
                    epoch_no_update += 1 
                    
                if epoch_no_update >= settings.patience: break

            # Load best model parameters after training
            model.load_state_dict(best_model_param)
            valid_loss, valid_references, valid_hypotheses = self.epoch_eval(model, self.valid_loader)

            # Calculate validation metrics 
            valid_bleu = corpus_bleu(valid_references, valid_hypotheses)
            valid_loss = np.mean(valid_loss)

            trial.set_user_attr('best_train_stats', (train_stats['train_loss'], train_stats['train_bleu']))
            trial.set_user_attr('best_es_stats', (train_stats['es_loss'], train_stats['es_bleu']))
            trial.set_user_attr('valid_loss', valid_loss)
            trial.set_user_attr('best_epoch_idx', best_epoch_idx)
            
            return valid_bleu
        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction = 'maximize')
        study.optimize(objective, n_trials = settings.n_trials)
        best_trial = study.best_trial
        # 
        self.best_hyperparams = best_trial.params
        self.best_hyperparams['best_epoch_num'] = best_trial.user_attrs['best_epoch_idx'] + 1

        # Save best hyperparameter configuration and its validation statistics
        torch.save(
            {
                "best_hyperparams" : self.best_hyperparams, 
                "train_stats" : best_trial.user_attrs['best_train_stats'], 
                "es_stats" : best_trial.user_attrs['best_es_stats'],
                "valid_stats" : best_trial.user_attrs['valid_loss'],
            }, 
            f"results/hpt_evaluation.pt"
        )

    def refit_model(self):
        best_hyperparams = self.best_hyperparams
        n_decoder_unfreeze = best_hyperparams['n_decoder_unfreeze']
        n_encoder_unfreeze = best_hyperparams['n_encoder_unfreeze']
        learning_rate = best_hyperparams['learning_rate']
        epoch_num = best_hyperparams['best_epoch_num']
        
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        model.to(self.device)

        set_trainable_layers(model, n_decoder_unfreeze, n_encoder_unfreeze)

        optimiser = AdamW(filter(lambda x : x.requires_grad, model.parameters()), lr=learning_rate)
        total_steps = epoch_num * len(self.train_loader)
        scheduler = get_linear_schedule_with_warmup(optimiser, num_warmup_steps=0, num_training_steps=total_steps)

        epoch_stats = {"train_loss" : [], "train_bleu" : []}
        for _ in range(epoch_num):
            train_losses, train_references, train_hypotheses = self.epoch_train(model, self.full_train_loader, optimiser, scheduler)
            train_bleu = corpus_bleu(train_references, train_hypotheses)
            epoch_stats["train_bleu"].append(train_bleu)
            epoch_stats['train_loss'].append(np.mean(train_losses))

        test_loss, test_references, test_hypotheses = self.epoch_eval(model, self.test_loader)

        # Calculate test metrics 
        test_bleu = corpus_bleu(test_references, test_hypotheses)
        test_loss = np.mean(test_loss)

        final_model_params = copy.deepcopy(model.state_dict())
        # Save parameters of the model and refitting statistics
        torch.save(
            {
                "final_model_param" : final_model_params, 
                "train_stats" : epoch_stats, 
                "test_stats" : {"test_loss" : test_loss, "test_bleu" : test_bleu}
            }, 
            f"results/refit_evaluation.pt"
        )
