import tensorflow as tf

from algorithms.encoder import transformer_encoder
def generator_encoder_producer(embedding_layer,projector,g_transformer):
    def encode(inputs):
        projected = projector(embedding_layer.embeddings)
        embedded = tf.nn.embedding_lookup(projected,inputs)
        encoded = g_transformer(embedded)
        return projected,encoded
    return encode
def discriminitor_encoder_producer(embedding_layer,d_transformer):
    def encode(inputs):
        embeded = embedding_layer(inputs)
        encoded = d_transformer(embeded)
        return encoded
    return encode



def build_graph(vocab_size,embedding_size,generator_size,
                    gn_blocks,gseq_length,gn_heads,gff_filter_size,g_dev,
                    dn_blocks,dseq_length,dn_heads,dff_filter_size,d_dev,mask_index,
                d_factor,learning_rate):
    g = tf.Graph()
    tf.reset_default_graph()
    with g.as_default():
        access_dict = {}
        access_dict["input"] = tf.placeholder(tf.int32,shape=[None,gseq_length],name="input")
        access_dict["training"] = tf.placeholder(tf.bool,shape=[],name="training")
        access_dict["peep"] = tf.placeholder(tf.bool,shape=[],name="peep")
        access_dict["mask_values"] = tf.placeholder(tf.int32,shape=[None,None],name="mask_values")
        access_dict["position_indeces"] = tf.placeholder(tf.int32,shape=[None,None,None],name="position_indeces")
        access_dict["target_word_indeces"] = tf.placeholder(tf.int32,shape=[None,None,None],name="target_word_indeces")
        
        with tf.device("/CPU:0"):
            embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size)
            embedding_layer(tf.constant([0]))
            embedding_projector = tf.keras.layers.Dense(generator_size)
        with tf.device(g_dev):
            g_transformer = transformer_encoder(gn_blocks,gseq_length,generator_size,
                                                gn_heads,gff_filter_size,name="gt")
            generator_encoder = generator_encoder_producer(embedding_layer,embedding_projector,g_transformer)
        with tf.device(d_dev):
            d_transformer = transformer_encoder(dn_blocks,dseq_length,embedding_size,
                                                dn_heads,dff_filter_size,name="dt")
            discriminitor_encoder = discriminitor_encoder_producer(embedding_layer,d_transformer)
            output_layer = tf.keras.layers.Dense(1,activation="sigmoid")
        losses,layers = {"generator_loss":0,"discriminitor_loss":0},{}

        if access_dict["training"]==False:
            d_encoded = discriminitor_encoder(access_dict["input"])
            layers["d_encoded"] = d_encoded
        else:
            corrupted = tf.tensor_scatter_nd_update(
                access_dict["input"],access_dict["position_indeces"],
                access_dict["mask_values"])
            if access_dict["peep"]==False:
                g_projected,g_encoded = generator_encoder(corrupted)
            else:
                g_projected,g_encoded = generator_encoder(access_dict["input"])
            masked_g_encoded = tf.gather_nd(g_encoded,access_dict["position_indeces"])
            generated = tf.nn.softmax(
                tf.transpose(
                    tf.matmul(g_projected,masked_g_encoded,transpose_b=True),[0,2,1]
                )
            )
            print(generated.shape)
            print("shit")
            losses["generator_loss"] = tf.reduce_sum(-tf.math.log(
                tf.gather_nd(generated,access_dict["target_word_indeces"])+1e-6))
            
            replaced = tf.tensor_scatter_nd_update(
                access_dict["input"],
                access_dict["position_indeces"],tf.cast(tf.math.argmax(generated,axis=-1),tf.int32)
            )
            labels = tf.cast(tf.clip_by_value(tf.abs(access_dict["input"]-replaced),0,1),tf.float32)
            target_signs = labels*-2+1

            d_encoded = discriminitor_encoder(replaced)
            layers["d_encoded"] = d_encoded
            d_out = output_layer(d_encoded)
            pre_d_loss = (tf.squeeze(d_out)-labels)*target_signs
            losses["discriminitor_loss"] = tf.reduce_sum(tf.math.log1p(pre_d_loss+1e-6))
        access_dict["outputs"] = layers["d_encoded"]        
        access_dict["losses"] = losses["generator_loss"]+d_factor*losses["discriminitor_loss"]
        access_dict["g_loss"] = losses["generator_loss"]
        access_dict["d_loss"] = losses["discriminitor_loss"]
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        access_dict["training_op"] = optimizer.minimize(access_dict["losses"])
        access_dict["init"] = tf.global_variables_initializer()
        print("encoded",access_dict["outputs"].shape)
    return g, access_dict