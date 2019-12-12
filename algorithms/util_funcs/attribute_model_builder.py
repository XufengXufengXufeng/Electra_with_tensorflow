from ..encoder import transformer_encoder_producer
from .data_preprocessing import data_batcher
from .model_handler import save_models

import tensorflow as tf
import numpy as np
import pandas as pd
def build_graph(max_len,vocab_size, embedding_size,n_blocks,n_heads,ff_filter_size,
                token_label_size,intent_size,learning_rate):
    g = tf.Graph()
    tf.reset_default_graph()
    with g.as_default():
        access_dict = {}
        access_dict["input"] = tf.placeholder(tf.int32,shape=[None,max_len],name="input")
        access_dict["target"] = tf.placeholder(tf.int32,shape=[None],name="target")
        #access_dict["seq_length"] = tf.placeholder(tf.int32,shape=[None],name="seq_length")
        access_dict["token_labels"] = tf.placeholder(tf.float32,shape=[None,max_len],name="token_labels")
        embeddings = tf.keras.layers.Embedding(vocab_size, 
                                                 embedding_size)(access_dict["input"])
        transformer_encoder = transformer_encoder_producer(
            n_blocks,max_len,embedding_size,n_heads,ff_filter_size
        )
        encoded = transformer_encoder(embeddings)
        token_layers = tf.keras.layers.Dense(token_label_size,name="token_layers",activation="sigmoid")
        token_layers.trainable=False
        token_logits = tf.concat([tf.expand_dims(token_layers(encoded[:,i,:]),axis=1) 
                        for i in range(max_len)],axis=1)
        print("token_logits",token_logits)
        print("encoded",encoded)
        #print(1/0)
        #logits_agg = tf.keras.layers.Dense(intent_size,name="logits")(lstms[:,-1,:])
        token_labels_reshaped = tf.expand_dims(access_dict["token_labels"],axis=2)
        print("token_labels_reshaped",token_labels_reshaped.shape)
        token_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(
            target=token_labels_reshaped,output=token_logits
        ))
        gated_encode = encoded*token_logits
        print("gated_encode",gated_encode)
        flatten = tf.keras.layers.Flatten()(gated_encode)
        logits_agg = tf.keras.layers.Dense(intent_size,name="logits_agg")(flatten)
        target_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=access_dict["target"],logits=logits_agg
        ))
        access_dict["loss"] = token_loss+target_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        access_dict["training_op"] = optimizer.minimize(access_dict["loss"])
        correct = tf.nn.in_top_k(logits_agg,access_dict["target"],1)
        access_dict["accuracy"] = tf.reduce_mean(tf.cast(correct,tf.float16))
        access_dict["prediction"] = tf.math.argmax(logits_agg,dimension=-1,name="prediction")
        access_dict["init"] = tf.global_variables_initializer()
        access_dict["max_len"] = max_len
        print("embeddings",embeddings)
        print("encoded",encoded)
        print("flatten",flatten)
        print("logits_agg",logits_agg)
    return g, access_dict
def train_tf_model(graph,access_dict,ided_data,w2id,tf_model_name,batch_size,epochs=100):
    with tf.Session(graph=graph) as sess:
        
        access_dict["init"].run()
        losses = []
        #"""
        for e in  range(epochs):
            db = data_batcher(ided_data,access_dict["max_len"],batch_size=batch_size)
            tmp_losses = []
            for inputs,token_labels,target in db:
                #loss = xentropy()
                feed = {"input":inputs,"target":target,
                        "token_labels":token_labels}
                loss,_ = sess.run([access_dict["loss"],access_dict["training_op"]],
                                 feed_dict = {access_dict[k]:feed[k] for k in feed.keys()})
                tmp_losses.append(float(loss.mean()))
                losses.append(float(np.mean(tmp_losses)))
                tmp_losses = []
            acc =  access_dict["accuracy"].eval(feed_dict={access_dict[k]:feed[k] for k in feed.keys()})
            print("epoch {}: train_accuracy is {:.2f};".format(e,acc))
#         save_models(sess,access_dict["input"],
#                     access_dict["prediction"],tf_model_name,w2id)
        save_models(sess,tf_model_name,w2id)
        #file_writer = tf.summary.FileWriter('logs', sess.graph)
    pd.Series(losses).plot();