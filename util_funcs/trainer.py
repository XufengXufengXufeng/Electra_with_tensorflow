import numpy as np
import tensorflow as tf

from util_funcs.after_training import save_models
def train_tf_model(graph,access_dict,data_batcher,tf_model_name,epochs=5):
    with tf.Session(graph=graph) as sess:
        
        access_dict["init"].run()
        losses,g_losses,d_losses = [],[],[]
        #"""
        for e in  range(epochs):
            db = data_batcher()
            tmp_losses,tmp_g_losses,tmp_d_losses,mask_counts = [],[],[],[]
            mask_count = 0
            end_flag = False
            i = 0
            while not end_flag:
                end_flag,padded,position_indeces,target_word_indeces,mask_values = db()
                if (len(position_indeces[0])==0)|(len(target_word_indeces[0])==0):
                    continue
                feed = {"input":padded,"position_indeces":position_indeces,
                        "target_word_indeces":target_word_indeces,"mask_values":mask_values,
                       "training":True}
                loss,g_loss,d_loss,_ = sess.run([
                    access_dict["losses"],
                                   access_dict["g_loss"],access_dict["d_loss"],access_dict["training_op"]],
                                 feed_dict = {access_dict[k]:feed[k] for k in feed.keys()})
                
                tmp_losses.append(float(loss.mean()))
                tmp_g_losses.append(float(g_loss.mean()))
                tmp_d_losses.append(float(d_loss.mean()))
                mask_counts.append(len(mask_values[0]))
                if i%52==0:
                    losses.append(float(np.mean(tmp_losses)))
                    g_losses.append(float(np.mean(tmp_g_losses)))
                    d_losses.append(float(np.mean(tmp_d_losses)))
                    mask_count = np.mean(mask_counts)
                    tmp_losses,tmp_g_losses,tmp_d_losses,mask_counts = [],[],[],[]
                    print("epoch {}: loss is {:.2f} g_loss is {:.2f} d_loss is {:.2f} mask count is {:.2f}"
                              .format(e,losses[-1],g_losses[-1],d_losses[-1],mask_count))
                i += 1
            db(True);
#         save_models(sess,access_dict["input"],
#                     access_dict["prediction"],tf_model_name,w2id)
            save_models(sess,tf_model_name+"_epoch_{}".format(e),access_dict)
        #file_writer = tf.summary.FileWriter('logs', sess.graph)
#             pd.Series(losses).clip(None,10).plot();
#             plt.show();
#             pd.Series(g_losses).clip(None,5).plot();
#             plt.show();
#             pd.Series(d_losses).clip(None,2).plot();
#             plt.show();