"""
#TODO change this comment to something describing your project
Project 2

At the end you should see something like this
Step Count:300
Training accuracy: 0.880000 loss: 0.444277
Test accuracy: 0.620000 loss: 1.418351

play around with your model to try and get an even better score
"""

import tensorflow as tf
import dataUtils

TENSORBOARD_LOGDIR = "logdir"

# Clear the old log files
dataUtils.deleteDirectory(TENSORBOARD_LOGDIR)


### Build tensorflow blueprint ###
# Tensorflow placeholder
input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32 ,3])

# View sample inputs in tensorboard
tf.summary.image("input_image", input_placeholder)

# Normalize image
# Subtract off the mean and divide by the variance of the pixels.
normalized_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), input_placeholder)

#conv layers

final_conv_layer = tf.layers.conv2d(normalized_image,
                                    filters=10,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same',
                                    activation=tf.nn.relu)


# convert 3d image to 1d tensor (don't change batch dimension)
flat_tensor = tf.contrib.layers.flatten(final_conv_layer)

#TODO improve fully connected layers
## Neural network hidden layers

hidden_layer_1 = tf.nn.dropout(tf.layers.dense(tf.layers.batch_normalization(flat_tensor, training=True),
                                              113, activation=tf.nn.relu), keep_prob=0.9)

## Logit layer
logits = tf.layers.dense(hidden_layer_1, 10)


# label placeholder
label_placeholder = tf.placeholder(tf.uint8, shape=[None])
label_one_hot = tf.one_hot(label_placeholder, 10)

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_one_hot, logits=logits))

#TODO choose better backpropagation
# backpropagation algorithm
train = tf.train.AdamOptimizer().minimize(loss)

accuracy = dataUtils.accuracy(logits, label_one_hot)

# summaries
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', loss)

tf.summary.tensor_summary("logits", logits)
tf.summary.tensor_summary("labels", label_one_hot)
summary_tensor = tf.summary.merge_all()


saver = tf.train.Saver()


## Make tensorflow session
with tf.Session() as sess:
    training_summary_writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR + "/training", sess.graph)
    test_summary_writer = tf.summary.FileWriter(TENSORBOARD_LOGDIR + "/test" , sess.graph)

    ## Initialize variables
    sess.run(tf.global_variables_initializer())


    step_count = 0
    while True:
        step_count += 1

        # get batch of training data
        batch_training_data, batch_training_labels = dataUtils.getCIFAR10Batch(is_eval=False, batch_size=50)

        # train network
        training_accuracy, training_loss, summary,  _ = sess.run([accuracy, loss, summary_tensor, train], feed_dict={input_placeholder: batch_training_data,
                                                                         label_placeholder: batch_training_labels})

        # write data to tensorboard
        training_summary_writer.add_summary(summary, step_count)

        # every 10 steps check accuracy
        if step_count % 10 ==  0:
            # get Batch of test data
            batch_test_data, batch_test_labels = dataUtils.getCIFAR10Batch(is_eval=True, batch_size=50)

            # do eval step to test accuracy
            test_accuracy, test_loss, summary = sess.run([accuracy, loss, summary_tensor], feed_dict={input_placeholder: batch_test_data,
                                             label_placeholder: batch_test_labels})

            # write data to tensorboard
            test_summary_writer.add_summary(summary, step_count)

            print("Step Count:{}".format(step_count))
            print("Training accuracy: {:.6f} loss: {:.6f}".format(training_accuracy, training_loss))
            print("Test accuracy: {:.6f} loss: {:.6f}".format(test_accuracy, test_loss))


        if step_count % 100 == 0:
            save_path = saver.save(sess, "model/model.ckpt")

        # stop training after 1,000 steps
        if step_count > 10000:
            break
