"""
Bert Trainer Module
Summary: Module for training and saving transformer model for multiple languages

Usage:
        >> from trainer_transformers import TransformerTrainer
        >> transformer_trainer = TransformerTrainer(
            pretrained_path="bert_base_uncased_pretrained/"
        )

        # Train model
        >> transformer_trainer.train(
            sentences = sentences,
            labels = labels,
            batch_size = batch_size,
            epochs = epochs
        )

        # Save Model
        >> transformer_trainer.save(
            model_dir = 'pytorch_model/' # example
        )
        # -- OR --
        >> transformer_trainer.save_onnx(
            model_dir = 'onnx_model/' # example
        )

ISSUES: 
    ONNX Quantization doesn't work for LaBSE model
WARNINGS:
    Libraries which are not fork safe might cause issue while using Trainer class with celery.
    To handle this TransformerTrainer can be imported inside a celery task function


Requirements - Python 3.8 (Tested upon):
    transformers==4.8.2
    torch==1.9.0
    onnx==1.9.0
    onnxruntime==1.8.1
    onnxruntime-tools==1.7.0
    numpy
    scikit-learn

Author: Swapnil Masurekar
"""

import os, shutil
import time
import datetime
import random
import pickle
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np

import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sklearn.preprocessing import LabelEncoder

# Training
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model Size optimization/quantization with onnx
from transformers.convert_graph_to_onnx import convert, optimize


# Set the seed value all over the place to make training reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)





def flat_accuracy(preds, labels):
    """Function to calculate the accuracy of our predictions vs labels"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss"""
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

    



class TransformerTrainer():
    """
        Transformer Trainer Class for sequence classification
        Class Usage:
            bert_trainer = TransformerTrainer(lang)
            bert_trainer.train(X, Y)
            bert_trainer.save(path) OR bert_trainer.save_onnx(path)
    """

    # Maxlen for padding and truncation
    max_len = 64

    train_val_split = 0.98 # % of train data

    onnx_conversion_opset = 11

    def __init__(
            self,
            # Model parameters
            pretrained_path:str, # Uses same path for pretrained model and tokenizer

        ):
        self.pretrained_path = pretrained_path

        # Loading pretrained tokenizer --
        print('Loading pretrained tokenizer for Transformer ...')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        print('Loaded pretrained tokenizer for Transformer')

        self.model_created = False


        # Changes will be needed in this function to incorporate multi-gpu training
        def assign_device():
            """Assign availale device"""
            # If there's a GPU available...
            print('Assigning Device ...')
            if torch.cuda.is_available():    

                # Tell PyTorch to use the GPU.    
                device = torch.device("cuda")

                print('There are %d GPU(s) available.' % torch.cuda.device_count())

                print('We will use the GPU:', torch.cuda.get_device_name(0))

            # If not...
            else:
                print('No GPU available, using the CPU instead.')
                device = torch.device("cpu")
            return device
        
        self.device = assign_device()





    # >> API METHOD: # Train Method
    def train(
        self,
        sentences:list,
        labels:list,
        batch_size:int=32,
        epochs:int=10
        ):
        """
        Args:
            sentences: list of preprocessed sentences using TransformerPreprocess class with same model type
            labels: list of labels

        """

        # ----------
        # Data Preparation
        # Encode labels using label encoder
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(labels)


        # Preparing input data for model
        input_ids, attention_masks = self.get_input_ids_attention_mask(self.tokenizer, self.max_len, sentences)
        labels = torch.tensor(labels)


        # Train Validation split
        train_dataset, val_dataset = self.perform_train_val_split(input_ids, attention_masks, labels, self.train_val_split)


        # Create Data loaders for train and validation
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly. 
                    batch_size = batch_size # Trains with this batch size.
                )
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )
        


        # ----------
        # Training Model

        # Loading pretrained model --
        print('Loading pretrained model for Transformer ...')
        self.model = AutoModelForSequenceClassification.from_pretrained( 
                self.pretrained_path,
                num_labels = len(self.label_encoder.classes_), 
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False, # Whether the model returns all hidden-states.
                ignore_mismatched_sizes=True
            )
        print('Loaded pretrained model for Transformer')
        self.model_created = True


        # Tell pytorch to run this model on the available device.
        if self.device.type in ['gpu', 'cuda']:
            self.model.to(self.device)
            print('Loaded pretrained model for Transformer on GPU')


        training_stats = self.train_model(
                        train_dataloader = train_dataloader,
                        validation_dataloader= validation_dataloader,
                        epochs=epochs
                        )



    # >> API METHOD: Save Pytorch Model
    def save(
        self,
        model_dir:str
        ): # -> Tokenizer files, label_encoder and model files will be saved in model_dir/
        """
            Save model files, tokenizer files and label encoder (label_encoder.pickle) at model_dir
        """
        assert self.model_created, 'Please train the model before saving'
        try:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            print("Saving model to %s" % model_dir)

            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(model_dir)

            # Save tokenizer
            self.tokenizer.save_pretrained(model_dir)

            # Save label_encoder
            pickle.dump(self.label_encoder, open(os.path.join(model_dir, 'label_encoder.pickle'), 'wb'))

        except Exception as e:
            raise Exception(e)


    # >> API METHOD: # Save ONNX Optimized and Quantized Model
    def save_onnx(
        self,
        model_dir:str
        ): # -> model will be saved as 'model_dir/onnx-model-quantized.onnx' along with Tokenizer files and label_encoder
        """
            !!ISSUE!!: Quantization doesn't work for LaBSE model yet

            Optimize and Quantize model first. And then save the optimized-quantized model
            Save model at model_dir with name: "onnx-model-quantized.onnx"

            Steps:
                -> This function creates an TEMPORARY folder 
                -> Saves the pytorch model in that folder
                -> Saves ONNX converted model in that folder 
                -> Saves Optimized model in that folder
                -> Save Quantized model at specified path 
                -> Deletes TEMPORARY folder

            Args:
                optimize:bool - True: optimize model during conversion, set False to skip optimize graph step
                quantize:bool - True: quantize model during conversion, set False to skip quantize graph step
        """
        assert self.model_created, 'Please train the model before saving'
        # Creating temporary directory
        tmpdirname = tempfile.mkdtemp(prefix="save_onnx_"+str(uuid4())+"__")
        print('Temporary directory created at:', tmpdirname)

        tokenizer_and_label_encoder_dir = os.path.join(model_dir, 'tokenizer_and_label_encoder/')
        try:
            if not os.path.exists(tokenizer_and_label_encoder_dir):
                os.makedirs(tokenizer_and_label_encoder_dir)

            # Save tokenizer
            self.tokenizer.save_pretrained(tokenizer_and_label_encoder_dir)

            # Save label_encoder
            pickle.dump(self.label_encoder, open(os.path.join(tokenizer_and_label_encoder_dir, 'label_encoder.pickle'), 'wb'))


            torch_model_dir = os.path.join(tmpdirname, 'torch_model_dir/')
            onnx_model_dir = os.path.join(tmpdirname, 'onnx_model_dir/')
            os.makedirs(torch_model_dir)
            os.makedirs(onnx_model_dir)
            print('Created temporary Directories for storing pt and onnx models:', torch_model_dir, onnx_model_dir)

            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(torch_model_dir)
            self.tokenizer.save_pretrained(torch_model_dir)

            convert(
                framework = 'pt',
                model = torch_model_dir,
                output = Path(os.path.join(onnx_model_dir, 'model.onnx')).absolute(),
                opset = self.onnx_conversion_opset,
                pipeline_name = 'sentiment-analysis'
            )
            print("Temporary ONNX converted model saved")

            optimized_model_path = optimize(Path(os.path.join(onnx_model_dir, 'model.onnx')).absolute())
            print("Temporary ONNX optimized model saved")

            quantized_model_path = self.quantize(
                onnx_model_path=optimized_model_path,
                quantized_model_path=Path(os.path.join(model_dir, 'onnx-model-quantized.onnx'))
                )
            print("ONNX quantized model saved at:", os.path.join(model_dir, 'onnx-model-quantized.onnx'))

            shutil.rmtree(tmpdirname)

        except Exception as e:
            shutil.rmtree(tmpdirname)
            raise Exception(e)
    

    @staticmethod
    def quantize(onnx_model_path: Path, quantized_model_path: Path) -> Path:
        """
        Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU

        Args:
            onnx_model_path: Path to location the exported ONNX model is stored

        Returns: The Path generated for the quantized
        """
        import onnx
        from onnxruntime.quantization import QuantizationMode, quantize

        onnx_model = onnx.load(onnx_model_path.as_posix())

        # Discussed with @yufenglee from ONNX runtime, this will be address in the next release of onnxruntime
        print(
            "As of onnxruntime 1.4.0, models larger than 2GB will fail to quantize due to protobuf constraint.\n"
            "This limitation will be removed in the next release of onnxruntime."
        )

        quantized_model = quantize(
            model=onnx_model,
            quantization_mode=QuantizationMode.IntegerOps,
            force_fusions=True,
            symmetric_weight=True,
        )

        # Save model
        print(f"Quantized model has been written at {quantized_model_path}: \N{heavy check mark}")
        onnx.save_model(quantized_model, quantized_model_path.as_posix())

        return quantized_model_path
    
    
    @staticmethod
    def get_input_ids_attention_mask(tokenizer, max_len:int, sentences:list):
        """
        Tokenize all of the sentences and map the tokens to thier word IDs.
        Args:
            tokenizer - Transformer tokenizer
            sentences - list of preprocessed sentences
        """
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = max_len,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                        )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return input_ids, attention_masks


    @staticmethod
    def perform_train_val_split(input_ids, attention_masks, labels, split_ratio:float):
        """Perform train val split"""
        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Calculate the number of samples to include in each set.
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset


    def train_model(
        self,
        train_dataloader,
        validation_dataloader,
        epochs:int
        ):
        """
            Main training Loop
            Reference: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
            Args:
                train_dataloader - Train DataLoader created from prepared model inputs
                validation_dataloader - Validation DataLoader created from prepared model inputs
            Returns:
                training_stats: list of dicts
        """

        # Initializing Optimizer
        optimizer = AdamW(self.model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)

        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, epochs):
            
            # ========================================
            #               Training
            # ========================================
            
            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train() # train mode

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because 
                # accumulating the gradients is "convenient while training RNNs". 
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()        

                # Perform a forward pass (evaluate the model on this training batch).
                # In PyTorch, calling `model` will in turn call the model's `forward` 
                # function and pass down the arguments. The `forward` function is 
                # documented here: 
                # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
                # The results are returned in a results object, documented here:
                # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
                # Specifically, we'll get the loss (because we provided labels) and the
                # "logits"--the model outputs prior to activation.
                result = self.model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels,
                            return_dict=True)

                loss = result.loss
                logits = result.logits

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)            
            
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))
                
            if len(validation_dataloader.dataset) > 0:
                # ========================================
                #               Validation
                # ========================================
                # After the completion of each training epoch, measure our performance on
                # our validation set.

                print("")
                print("Running Validation...")

                t0 = time.time()

                # Put the model in evaluation mode--the dropout layers behave differently
                # during evaluation.
                self.model.eval()

                # Tracking variables 
                total_eval_accuracy = 0
                total_eval_loss = 0
                nb_eval_steps = 0

                # Evaluate data for one epoch
                for batch in validation_dataloader:
                    
                    # Unpack this training batch from our dataloader. 
                    #
                    # As we unpack the batch, we'll also copy each tensor to the GPU using 
                    # the `to` method.
                    #
                    # `batch` contains three pytorch tensors:
                    #   [0]: input ids 
                    #   [1]: attention masks
                    #   [2]: labels 
                    b_input_ids = batch[0].to(self.device)
                    b_input_mask = batch[1].to(self.device)
                    b_labels = batch[2].to(self.device)
                    
                    # Tell pytorch not to bother with constructing the compute graph during
                    # the forward pass, since this is only needed for backprop (training).
                    with torch.no_grad():        

                        # Forward pass, calculate logit predictions.
                        # token_type_ids is the same as the "segment ids", which 
                        # differentiates sentence 1 and 2 in 2-sentence tasks.
                        result = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    return_dict=True)

                    # Get the loss and "logits" output by the model. The "logits" are the 
                    # output values prior to applying an activation function like the 
                    # softmax.
                    loss = result.loss
                    logits = result.logits
                        
                    # Accumulate the validation loss.
                    total_eval_loss += loss.item()

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()

                    # Calculate the accuracy for this batch of test sentences, and
                    # accumulate it over all batches.
                    total_eval_accuracy += flat_accuracy(logits, label_ids)
                    

                # Report the final accuracy for this validation run.
                avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
                print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

                # Calculate the average loss over all of the batches.
                avg_val_loss = total_eval_loss / len(validation_dataloader)
                
                # Measure how long the validation run took.
                validation_time = format_time(time.time() - t0)
                
                print("  Validation Loss: {0:.2f}".format(avg_val_loss))
                print("  Validation took: {:}".format(validation_time))

                # Record all statistics from this epoch.
                training_stats.append(
                    {
                        'epoch': epoch_i + 1,
                        'Training Loss': avg_train_loss,
                        'Valid. Loss': avg_val_loss,
                        'Valid. Accur.': avg_val_accuracy,
                        'Training Time': training_time,
                        'Validation Time': validation_time
                    }
                )
            else:
                # No validation
                training_stats.append(
                    {
                        'epoch': epoch_i + 1,
                        'Training Loss': avg_train_loss,
                        'Valid. Loss': 0,
                        'Valid. Accur.': 0,
                        'Training Time': training_time,
                        'Validation Time': 0
                    }
                )

        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        return training_stats
        


if __name__ == '__main__':
    pass
