# Electra_with_tensorflow
This is an implementation of electra according to the paper {ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators}

# Things to know before you read the project:

1. This is a very raw project. Too rough to use for production. It's not well organized and tested, so not good for research either. It may just provide some ideas when you want to implement electra.

2. There are some differences between my implement and the original electra paper:

2.1 I don't have any powerful computing resources, so i haven't used matrix multiplication for masking. Simplly put, for each batch i use a randomed sample size. Each batch has same number of token being masked.

2.2 In the paper, the authors state the generator hidden size is half of the discriminator hidden size. I haven't figure out how to half the hidden size of a transformer encoder. To my understanding, halfing the hidden size is the same as halfing the number of heads. So I just half the head counts (in my hyper-parameters, it is not halved exactly, it just 4/6. I have no reason for the action. in fact i haven't done any effort on tuning the model at all.)

3. As you may probably tell, this project is not polished. There maybe some errors that I haven't found. I haven't used the datasets that were used in the paper. I used a chinese dataset, so there is no reference regarding how well the model should work on the dataset. All in all, it's not well experimented project, I suggest my fellow viewers don't dive too much in this project, if you want to make a production ready application.

4. My writing with tensorflow is not standard, as you may tell. I use many functional programming. I do this because I didn't read tensorflow user guide enough, also because I feel comfortable writing functions. I think data types more complex than the prime types are mutable and the tensorflow layers feel more complex than dictionary, so I just write functions with no test at all. That's probably why errors may happen running my project.

# how to run this project
## the environment 
I use tensorflow official image for version 1.14; with docker just 
> docker pull tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

## the data
the entrance to the program is data. I don't want to be crue, but you really have to write functions to format your data into **one tokenized sentence (token ids seperated with comma) per line txt file**. That's the train.txt format.

## the configure.yml
datafiles: 

 - "data/baike_qa_valid.json" # my raw valid data location. remember you should use your own data formater functions. 

- "data/baike_qa_train.json" # my raw train data location.

char2id_loc: "data/processed_data/char2id.json" # this is char2id file after formatting (data processing), this could be word2id, depending on how you tokenize your raw data.

id2char_loc: "data/processed_data/id2char.json" # this is id2char file after formatting

train_data_loc: "data/processed_data/train.txt" # this is the formatted train data. from here you can tell how rough this project and how lazy i am, as i don't even produce the valid data.

embedding_size: 100 # this is the embedding size

generator_size: 50 # this is the generator hidden size, which is also the discriminator hidden size.

gn_blocks: 1 # this the number of the generator transformer block.

seq_length: 512 # this is the max sequence length

gn_heads: 4 # this is generator head count.

gff_filter_size: 150 # this is generator feed forward filter size.

g_dev: "/CPU:0" # this is the device I use, I once had a GPU, but later i lost it.

dn_blocks: 3 # this is the number of the discriminator transformer block

dn_heads: 6 # this is the discriminator head count.

dff_filter_size: 300 # this is the discriminator feed forward filter size.

d_dev: "/CPU:0" # this is the same GPU loss story.

d_factor: 50 # this is the factor that is used to emp the discriminator loss.

learning_rate: 1e-3 # this is the learning rate.

max_len: 512 # this is the max sequence length again. This duplication is a result of my lazyness not a well thought action.
