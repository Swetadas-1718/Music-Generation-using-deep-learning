# Music Generation
<img src = https://github.com/Swetadas-1718/Music-Generation/blob/main/sheet%20music.jpg>

## Refer my medium article for the same: https://medium.com/mlearning-ai/music-generation-using-deep-learning-49692851c57c


## Real World Problem
- The objective of music generation is to explore deep learning regarding the field of music composition using artificial intelligence.
- The case study focuses on generating music automatically using Recurrent Neural Networks(RNN).
- We do not need to be an expert to generate music. Even non experts like me can generate a descent quality creative music using RNN.
- Creating music was something unique to humans until now, but some great advancements have been made using deep learning.

## Objective
- Building a model that takes existing music data as input learns the pattern and generates "new" music.
- The model-generated music need not be professional, as long as it is melodious and good to hear.
- It cannot simply copy and paste from the training data. It has to learn the patterns from the existing music that is enjoyed by humans.

## Now, what is music?
- Basically, music is a sequence of musical components/events.
- Input- Sequence of musical events/notes
- Output- New sequence of musical events/notes
- In this case study, I have limited myself to single instrument music. You can extend this to multiple instrument music.

## Music Representation
- Sheet music representation can be used for both single instrument and multi instrument.----> visual file
- Abc - notation---- popular
- MIDI---> popular
- Mp3 ---> audio files ----> actual audio file
In this case study, we will focus on abc notation as it is the simplest one and just uses alpha numeric character.

## Why MP3 is not considered as music representation?
- Mp3 contains frequency, amplitude, and timestamp.
- Musicians use ABC notation or sheet music as a representation which is much more efficient because they don't generate music by thinking in terms of frequency. 
- So, it's better to leverage thousands of years of music notation that phenomenal musicians have designed.
- Hence, we will not use mp3 files. We will compose our music in the space of notations.

## Char-RNN Model (High-Level Overview)
So, we have some domain knowledge by now and we need not be an expert. We'll now ground this in and know about char-RNN. Since our music is a sequence of characters, therefore our obvious choice will be RNNs.
- There is a special type of RNN called "Char-RNN".
- We will be using many to many RNN. Here, we will feed the RNN with our characters of the sequence one by one and it will out the next character in the sequence.

## Data Obtaining:
- Refer : http://abc.sourceforge.net/NMD
- It says "ABC version of the Nottingham Music Database" which contains over 1000 folk tunes stored in a special text format.
- Of course, it takes a lot of time to train the model with larger data like 1000 tunes. So I will use the jigs dataset which contains about 340 tunes.
- You get a txt file with multiple tunes here.
- Simply copy and paste into a txt file as input.txt.
- Each tune is having a meta data section and music section.

## Data Preprocessing:
We want to preprocess the input.txt file into such a format that we can feed it into the RNN because the way we build our dataset will impact the model heavily and RNNs can be tricky.

## Model Architecture and training
<img src = https://github.com/Swetadas-1718/Music-Generation/blob/main/1_vYWoAafJRbkXBYXwjhckbQ.png>
- X is a matrix of (BATCH_SIZE,SEQ_LENGTH) = (16,64)
- Y is a 3D tensor of (BATCH_SIZE,SEQ_LENGTH,vocab_size) = (16,64,86). The vocab size is considered because of one-hot encoding.
- After embedding, (BACTH_SIZE,SEQ_LENGTH,embedding_dim) = (16,64,512)
- Now, we want to predict the next character which should be one of the 86 unique characters. So, it's a multi-class classification problem. Therefore, our last layer is the softmax layer of 86 activations.
- So, I will generate each of my batches and train them. For every training epoch, I will print the categorical cross-entropy loss and accuracy.
- I have 1,904,214 total parameters.
- As we are having so many parameters, so we are using dropouts with a keep probability of 0.2.
- By the time we reach 100 epochs while training, roughly around 90% + times, the model is able to predict what the next character is. So, our model is doing a pretty good job.
- At the end of 10 epochs, we are storing the weights of the model. We will use these weights to reconstruct the model and predict.

## Music Generation 
By now, our model got ready to predict.

## Further Scope
Well, we got pretty good results, but we can improve our model by training it with more tunes of multi-instruments.
Here, I trained my model with just 350 tunes. So, we can expose our model to more instruments and varieties of musical tunes, which will result in more melodious tunes with varieties.

## Built with
- ipython-notebook - Python Text Editor
- numpy, scipy- number python library
- pandas - data handling library
- Keras - Deep Learning Library

## Author
Swetapadma Das - Complete Work
