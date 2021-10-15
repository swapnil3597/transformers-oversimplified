"""
Predictor Transformers Module
Summary: 
 - Module to perform prediction on models trained and saved using TransformerTrainer
 - This Module performs prediction on the query preprocessed using TransformerPreprocess

Usage:
    >> transformer_predictor = TransformerPredictor(force_cpu=False)

    >> transformer_predictor.load_onnx('onnx_model/')
            OR
    >> transformer_predictor.load('pytorch_model/')

    >> pred_label, prob = transformer_predictor.predict(query)

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

import os
import pickle
import numpy as np

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers


# TODO: Make torch number of threads configurable
torch.set_num_threads(1) # Single threaded process to speed up the execution

class TransformerPredictor():

    # Maxlen for padding and truncation
    max_len = 64

    def __init__(
        self,
        force_cpu=False,
        ):

        self.pt_model_loaded = False
        self.onnx_model_loaded = False

        # Changes will be needed in this function to incorporate multi-gpu training
        def assign_device():
            """Assign availale device"""
            # If there's a GPU available...
            print('Assigning Device ...')
            if torch.cuda.is_available() and force_cpu==False:    
                device_index = 0 # default GPU device index
                # Tell PyTorch to use the GPU.    
                device = torch.device("cuda:{}".format(str(device_index))) # eg: "cuda:0", "cuda:1", ...

                print('There are %d GPU(s) available.' % torch.cuda.device_count())

                print('We will use the GPU:', torch.cuda.get_device_name(device_index))

            # If not...
            else:
                print('No GPU available, using the CPU instead.')
                device = torch.device("cpu")
            return device
        
        self.device = assign_device()
    
    
    # API Method: load label encoder, tokenizer and pytorch model
    def load(
        self,
        model_dir:str
        ):
        """
            Load label encoder, tokenizer and pytorch model
        """
        assert not self.onnx_model_loaded, "ONNX model is already loaded, can't load PyTorch model," +\
                                           "use different object instance to load PyTorch model"


        # Loading tokenizer and label_encoder --
        self.tokenizer, self.label_encoder = self.load_tokenizer_and_label_encoder(model_dir)      

        
        # Loading model --
        print('Loading model for Transformer ...')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        print('Loaded PyTorch model for Transformer Predictor')
        self.model_created = True

        # Tell pytorch to run this model on the available device.
        if self.device.type in ['gpu', 'cuda']:
            self.model.to(self.device)
            print('Loaded PyTorch model for Transformer Predictor on GPU')
        self.pt_model_loaded = True
        
    
    
    # API Method: load label encoder, tokenizer and ONNX model
    def load_onnx(
        self,
        model_dir:str
        ):
        """
            Load label encoder, tokenizer and onnx model
            WARNING: For now only CPUExecutionProvider supported
        """
        assert not self.onnx_model_loaded, "PyTorch model is already loaded, can't load ONNX model," +\
                                           "use different object instance to load ONNX model"

        # Helper functions ----
        def create_model_for_provider(model_path: str, provider: str) -> InferenceSession: 
            assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

            # Few properties that might have an impact on performances (provided by MS)
            options = SessionOptions()
            options.intra_op_num_threads = 1
            options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

            # Load the model as a graph and prepare the CPU backend 
            session = InferenceSession(model_path, options, providers=[provider])
            session.disable_fallback()
                
            return session
        
        
        tokenizer_and_label_encoder_dir = os.path.join(model_dir, 'tokenizer_and_label_encoder/')
        # Loading tokenizer and label_encoder --
        self.tokenizer, self.label_encoder = self.load_tokenizer_and_label_encoder(tokenizer_and_label_encoder_dir)      

        onnx_model_path = os.path.join(model_dir, 'onnx-model-quantized.onnx')
        self.onnx_model = create_model_for_provider(onnx_model_path, "CPUExecutionProvider")
        print('Loaded ONNX model for Transformer Predictor')
        self.onnx_model_loaded = True
    


    # API Method: To make predictions on a query
    def predict(self, query:str):
        """
            Make the prediction on the query using loaded model and return the results
            Args:
                query: str - preprocessed query using TransformerPreprocess() class
            
            Returns:
                label: str - predicted label
                prob: float - predicted label probabiltiy
        """
        assert self.pt_model_loaded or self.onnx_model_loaded, "Model not loaded, Kindly load the model for prediction"
        assert self.pt_model_loaded != self.onnx_model_loaded, "Both PyTorch and ONNX model shouldn't be loaded at the same time"
        
        encoded_dict = self.tokenizer.encode_plus(
                            query,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = self.max_len,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        if self.pt_model_loaded:
            
            input_ids = encoded_dict['input_ids']
            attention_mask = encoded_dict['attention_mask']

            # Loading inputs to device: cpu or gpu
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # Run PyTorch model inference
            result = self.model(input_ids, 
                                token_type_ids=None, 
                                attention_mask=attention_mask,
                                return_dict=True)
            logits = result.logits
            probs = torch.nn.functional.softmax(logits)
            logits = logits.detach().cpu().numpy()
            probs = probs.detach().cpu().numpy()

            pred_label_i = np.argmax(logits, axis=1).flatten() # Argmax
            pred_label = self.label_encoder.inverse_transform(pred_label_i)[0]
            prob = probs[0][pred_label_i][0]

        if self.onnx_model_loaded:
            inputs_onnx = {k: v.cpu().detach().numpy() for k, v in encoded_dict.items()}

            # Run ONNX model inference
            logits, = self.onnx_model.run(None, inputs_onnx)
            probs = torch.nn.functional.softmax(torch.from_numpy(logits))
            probs = probs.detach().cpu().numpy()

            pred_label_i = np.argmax(logits, axis=1).flatten() # Argmax
            pred_label = self.label_encoder.inverse_transform(pred_label_i)[0]
            prob = probs[0][pred_label_i][0]
        
        return pred_label, float(prob)
    
    
    def load_tokenizer_and_label_encoder(self, model_dir):
        """Load tokenizer and label_encoder from model_dir"""
        
        # Loading tokenizer --
        print('Loading tokenizer for Transformer Predictor...')
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print('Loaded tokenizer for Transformer Predictor')

        # Loading Label encoder --
        label_encoder = pickle.load(open(os.path.join(model_dir, 'label_encoder.pickle'), 'rb'))
        return tokenizer, label_encoder
