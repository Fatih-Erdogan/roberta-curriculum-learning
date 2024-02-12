# COMP442-FinalProject

The provided notebook contains the code necessary to create a language model
based on a specified data, using curriculum learning. 

### IMPORTANT NOTE:

The file paths in the project code need to be adjusted if you want to run the code.
Necessary explanations are provided next to these lines.

Here is the necessary explanation regarding the functions defined in the
notebook:

### get_tokenizer():
This function receives the phase as its input and constructs the tokenizer 
based on the hardcoded vocabulary sizes, using the same data files for each phase
to ensure the vocabulary consistency between each phase.

### init_model_and_tokenizer():
Receives the phase as the argument and returns the model whose parameters are
determined according to the phase, together with the tokenizer.

### merge_typeN():
Used for calculating the parameters of the next model based on the previous model.
Type 4 is specifically used for calculating the embedding and 
decoder matrices. The details are provided in the paper.

### prepare_embedding_and_out_layer():
Using merge_typeN() functions, calculates and sets the "embedding" and "lm_head"
parameters of the new RobertaForMaskedLM model based on the model from previous phase.

### prepare_encoder_layer():
Using merge_typeN() functions, iterates over the stacked encoder layers,
calculates and sets the "encoder" parameters of the
new RobertaForMaskedLM model based on the model from previous phase.

### prepare_model_for_new_phase():
Interface for the user to adjust the parameters of the new model.
Calls prepare_encoder_layer() and prepare_embedding_and_out_layer() functions.

### preprocess_file():
Taking the current tokenizer as input, splits the lines which are longer than a
hardcoded number of tokens, constructs a new datafile with new lines, returns the
new path

### train_model():
Trains a given model on the specified training files for the MLM task. Saves the
model after each epoch if the file path is provided.


## USAGE:

- Initialize the first model and tokenizer
- The first model is not passed to prepare_model_for_new_phase() because this is the
first model to be trained.
- Train the first model with specified training files (for phase 1)
- Initialize the second model and its tokenizer.
- Prepare the second model for new phase using the previous model (first model)
- Train the second model with specified training files (for phase 2)
- Initialize the third model and its tokenizer.
- Prepare the third model for new phase using the previous model (second model)
- Train the third model

In the provided code, after being trained, models are saved to a desired location
for further use.

Check the notebook and the paper for more details.
