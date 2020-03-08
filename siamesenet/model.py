import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    BatchNormalization, MaxPool2D, ReLU
from tensorflow.keras import Model


class SiameseNet(Model):

    def __init__(self):
        super(SiameseNet, self).__init__()
        # self.width, self.height, self.channel = w, h, c
        # self.classes = classes

        self.encoder = tf.keras.Sequential([
            Conv2D(input_shape=(None, None, 3), filters=64, kernel_size=5,      # Converted to grayscale
                   padding='valid', data_format="channels_last"),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2,2)),

            Conv2D(filters=128, kernel_size=5, padding='valid'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2,2)),

            Conv2D(filters=256, kernel_size=3, padding='valid'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=512, kernel_size=3, padding='valid'),
            BatchNormalization(),
            ReLU(),

            Flatten()
        ])
        # Fully connected layer
        self.dense = tf.keras.Sequential([
            Dense(1, activation=None)
        ])
        # print(self.encoder.weights)

    def prediction(self, input1, input2):
        batch_size = input1.shape[0]
        # fusion 2 inpusts into 1 and then encode
        all_input = tf.concat([input1, input2], 0)
        all_output = self.encoder(all_input)

        # split the output
        output1 = all_output[:batch_size]
        output2 = all_output[batch_size:]
        distance = tf.pow((output2 - output1), 2)  # Squared distance
        score = self.dense(distance)

        # labels_predict = tf.argmax(tf.nn.softmax(score), -1)
        # labels_predict = tf.argmax(tf.nn.sigmoid(score), -1)
        labels_predict = tf.cast(score>0, tf.float32)
        labels_predict = tf.squeeze(labels_predict)
        score = tf.squeeze(score)
        # print('score: ', score)
        # one_hot_labels = tf.one_hot(gt_labels, 2)

        # loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=score)
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_labels, logits=score)

        return labels_predict, score, all_output

    def loss(self, gt_labels, output1, output2):
        '''
        Samples same --> label=1, sample different --> label=0
        :return:
        '''
        distance = tf.pow((output2 - output1), 2)      #Squared distance
        score = self.dense(distance)
        # print('score: ', score)
        # one_hot_labels = tf.one_hot(gt_labels, 2)

        # loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=score)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_labels, logits=score)
        # labels_predict = tf.argmax(tf.nn.sigmoid(score), -1)

        return loss, labels_predict

        # loss = 0.5 * (tf.cast(gt_labels, tf.float32) * distance +
        #               tf.cast((1 + -1 * gt_labels), tf.float32) *
        #               tf.pow(tf.nn.relu(margin - (distance + eps)), 2))
        # print('Origin loss: ', loss)
        # return tf.reduce_mean(loss) if size_average else tf.reduce_sum(loss)

    # @tf.function
    def call(self, input1, input2, gt_labels):
        batch_size = input1.shape[0]
        #fusion 2 inpusts into 1 and then encode
        all_input = tf.concat([input1, input2], 0)
        all_output = self.encoder(all_input)

        #split the output
        output1 = all_output[:batch_size]
        output2 = all_output[batch_size:]

        #define the loss
        distance = tf.pow((output2 - output1), 2)  # Squared distance
        score = self.dense(distance)
        # print('score: ', score)
        # one_hot_labels = tf.one_hot(gt_labels, 2)

        score = tf.squeeze(score)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_labels, logits=score)
        # labels_predict = tf.argmax(tf.nn.sigmoid(score), -1)
        labels_predict = tf.cast(score>0, tf.float32)

        loss = tf.reduce_mean(loss)
        acc = tf.cast(tf.equal(labels_predict, gt_labels), tf.float32)
        acc = tf.reduce_mean(acc)

        # print(score)

        return loss, labels_predict, acc


    # def save(self, save_dir, model):
    #     checkpoint = tf.train.Checkpoint(model=model)
    #     checkpoint.save(file_prefix=checkpoint_prefix)
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     checkpoint_prefix = os.path.join(save_dir, 'model.ckpt')


    def load(self, load_dir, model):
        if not os.path.exists(load_dir):
            raise ValueError('Checkpoint file does not exist')
        latest = tf.train.latest_checkpoint(load_dir)
        checkppoint = tf.train.Checkpoint(model=model)
        checkppoint.restore(latest).assert_consumed()
