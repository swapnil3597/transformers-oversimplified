# Oversimplified Transformers Module

This is a transformers module with oversimplified usage for training an text classification problem build upon [Hugging Face Transformers](https://github.com/huggingface/transformers).

**Features:**
- No encoding of sentences is required, directly list of sentences and corresponding labels can form inputs
- Model can be directly saved as Pytorch model or ONNX graph with simple API and can be directly loaded for inference
- No need to maintain separate label encoder or tokenizer

## Acknowledgement:
The Training and Inference code is inspired from [this](https://mccormickml.com/2019/07/22/BERT-fine-tuning/) blog by Chris McCormick.

## Usage:

### Trainer Module:

```python
from trainer_transformers import TransformerTrainer

transformer_trainer = TransformerTrainer(
            pretrained_path="bert_base_uncased_pretrained/" 
            # This can be a pre-trained pytorch model path, 
            # OR Model ID from https://huggingface.co/transformers/pretrained_models.html
        )

transformer_trainer.train(
            sentences = sentences, # List of strings (train input sentences)
            labels = labels, # List of labels as string
            batch_size = batch_size, # integer value
            epochs = epochs # integer value
        )

# Save model as Pytorch model
transformer_trainer.save(
            model_dir = 'pytorch_model/' # output path example
        )
   # -- OR --
# Save model as ONNX graph
transformer_trainer.save_onnx(
            model_dir = 'onnx_model/' # output path example
        )
```


### Predictor Module:
```python
transformer_predictor = TransformerPredictor(force_cpu=False)

transformer_predictor.load_onnx('onnx_model/') # Incase .save_onnx() is used to save the model
            # -- OR --
transformer_predictor.load('pytorch_model/') # Incase .save() is used to save the model

pred_label, prob = transformer_predictor.predict(query) # query similar to string in sentences list
```
