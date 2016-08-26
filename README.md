# Dictionary Definition Models
A recurrent neural network that learns to define words from dictionaries.

## Dependencies
- [Torch](https://github.com/torch/torch7)
- [Python 2.7](https://www.python.org/) (for basic scripts)
- [Moses](http://www.statmt.org/moses/) (specifically we just use [sentence-bleu.cpp](https://github.com/moses-smt/mosesdecoder/blob/master/mert/sentence-bleu.cpp) for evaluation)

### CUDA Libraries
Skip this if you do not have a GPU.
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (v7.5)
- [cuDNN](https://developer.nvidia.com/cudnn) (R5, only for convolution network)

### Torch Libraries
Most of the libraries will come with Torch if you install from their installation script. You can use [luarocks](https://luarocks.org/) to install additional packages. For examples, ```luarocks install dp```. The additional packages are:
- [dp](https://github.com/nicholas-leonard/dp)
- [dpnn](https://github.com/Element-Research/dpnn)
- [rnn](https://github.com/Element-Research/rnn)

If you are planing to use GPU (CUDA), you will need the following packages:
- [cutorch](https://github.com/torch/cutorch)
- [cunn](https://github.com/torch/cunn)
- [cudnn](https://github.com/soumith/cudnn.torch) (make sure that you get the right branch for your cuDNN version)

### Word Embedding
You will also need a set of word embeddings in torch binary format of an object:
``` lua
{
  M, % 2D tensor where each row is an embedding
  v2wvocab, % index-to-word map
  w2vvocab % word-to-index map
}
```
You can download from [Word2Vec](https://code.google.com/archive/p/word2vec/) and use [word2vec.torch](https://github.com/rotmanmi/word2vec.torch) to convert them into torch binary file.
