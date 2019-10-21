# Interpretation as negative sampling

Basic code to run a simple reference task as negative sampling. Run with:

    python3 embed_model.py


The negative sampling implementation is contained in utils/ext2vec.py. For now, it uses a barebone implementation with a target matrix of words and a context matrix of image vectors. The task the network has to solve is to tell whether some image can be referred to as <word> or not. Positive and negative examples are given. 

The image matrix is frozen, ensuring image representations do not change in the course of training. The target word matrix, however, is randomly initialised, and is updated by backpropagation.

## Data

Some image vectors are contained in the *data/* directory, in the *55_vision_animals.txt* file. There is also a *file_word_mapping.txt* file, indicating the target word associated with each image. The file *55_vision_animals_mapping.bin* file contains a binary matrix showing which word (in the rows) is appropriate for which image (in the columns).


