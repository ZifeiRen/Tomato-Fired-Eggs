import tensorflow as tf
import numpy as np
import DataHandler as dh
from tensorflow.contrib.layers import batch_norm


class CNN(object):
    def __init__(self, input_dim=784, out_dim=10, lr=0.01, std=0.1):
        """
        Initialized Convolutional Neural Networks
        :param input_dim: input data dimension
        :param out_dim: Output dimensions
        :param lr: learning rate
        :param std: standard deviation
        """
        self.x_dim = input_dim
        self.out_dim = out_dim
        self.lr = lr
        self.std = std
        self._init_net()  # Initialized Convolutional Neural Network Structure
        self._init_train_methods()  # Initial Network Training Methods

    def _init_net(self):
        """
        Initialized Convolutional Neural Network Structure
        :return:
        """
        # Constructed placeholders
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.x_dim], name="input_data")  # [n,m]
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim], name="labels")
        # Reshaping the input tensor
        self.input_img = tf.reshape(self.x, shape=[-1, 28, 28, 1])
        # Constructing the first layer of convolution
        with tf.name_scope("layer_1"):
            layer_dim_1 = 50
            # w=[3,3,1,layer_dim_1] corresponds to layer_dim_one 3*3 convolutional kernel.
            w = self._init_varible(shape=[3, 3, 1, layer_dim_1], name='w')
            b = self._init_varible(shape=[layer_dim_1], name="b")
            conv = tf.nn.conv2d(input=self.input_img, filter=w, strides=[1, 1, 1, 1], padding="SAME") + b
            # Output Shape：[28,28,layer_dim1]
            conv = batch_norm(conv)  # Output Shape：[28,28,layer_dim1]
            active = tf.nn.relu(conv)  # Output Shape：[28,28,layer_dim1]
            # ksize=[1,2,2,1] Corresponds to 2*2 pooling
            active = tf.nn.max_pool(value=active, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # Output Shape：[14,14,layer_dim1]

        with tf.name_scope("layer_2"):
            layer_dim_2 = 50
            w = self._init_varible(shape=[3, 3, layer_dim_1, layer_dim_2], name='w')
            b = self._init_varible(shape=[layer_dim_2], name='b')
            conv = tf.nn.conv2d(input=active, filter=w, strides=[1, 1, 1, 1], padding="SAME") + b
            # Output Shape：[14,14,layer_dim2]
            conv = batch_norm(conv)
            active = tf.nn.relu(conv)
            active = tf.nn.max_pool(value=active, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # Output Shape：[7,7,layer_dim2]
        with tf.name_scope("layer_3"):
            layer_dim_3 = 25
            w = self._init_varible(shape=[3, 3, layer_dim_2, layer_dim_3], name='w')
            b = self._init_varible(shape=[layer_dim_3], name='b')
            conv = tf.nn.conv2d(input=active, filter=w, strides=[1, 1, 1, 1], padding="SAME") + b
            # Output Shape：[7,7,layer_dim_3]
            conv = batch_norm(conv)
            active = tf.nn.relu(conv)
        # Constructing the output layer
        with tf.name_scope("out_layer"):
            hidden_out = tf.reshape(active, shape=[-1, 7 * 7 * layer_dim_3])  # 将tensor重塑为[m行，n列]
            w = self._init_varible(shape=[7 * 7 * layer_dim_3, self.out_dim], name="w_out")
            b = self._init_varible(shape=[self.out_dim], name="b_out")
            self.out_scores = tf.matmul(hidden_out, w) + b

    def _init_train_methods(self):
        """
        Initial network training methods: loss function, gradient descent, session
        :return:
        """
        # Initialized loss function
        with tf.name_scope("cross_entropy"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out_scores, labels=self.y))
        # Construct correctness calculation
        with tf.name_scope("acc"):
            self._predict = tf.argmax(self.out_scores, 1)
            correct_predict = tf.equal(tf.argmax(self.out_scores, 1), tf.argmax(self.y, 1), name="correct_predict")
            self.acc = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        # Constructive gradient training methods
        with tf.name_scope("Adam"):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # initi Session
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=1)

    def train(self, data_dict, itrs=10000, batch_size=50):
        """
        Training Network
        :param data_dict: Training Data Dictionary
        :param itrs: number of training iterations
        :param batch_size: Sample data size
        :return:
        """
        for i in range(itrs):
            mask = np.random.choice(data_dict['train_x'].shape[0], batch_size, replace=True)
            batch_x = data_dict['train_x'][mask]
            batch_y = data_dict['train_y'][mask]
            # 训练模型
            self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y: batch_y})
            # 验证模型
            if i % 1000 == 0:
                tem_loss, tem_acc = self.test(data=data_dict['test_x'], labels=data_dict['test_y'])
                print("Iteration：", i, "times, current loss value：", tem_loss, " Current correctness：", tem_acc)
                self.save()
        self.sess.close()

    def test(self, data, labels):
        """
        Test Model Performance
        :param data: data eigenvalues
        :param labels: Target value of the data.
        :return: Returns the loss value and the correct percentage.
        """
        tem_loss, tem_acc = self.sess.run([self.loss, self.acc], feed_dict={self.x: data, self.y: labels})
        return tem_loss, tem_acc

    def save(self, path="model/cnn/"):
        """
         Save Model
        :param path:model save path
        :return:
        """
        self.saver.save(self.sess, save_path=path)

    def restore(self, path="model/cnn/"):
        """
        Recovery Model
        :param path: The path where the model is saved.
        :return:
        """
        self.saver.save(sess=self.sess, save_path=path)

    def predict(self, data):
        """
        Predicting input data target values
        :param data: Enter data characteristics.
        :return: Predict target value

        """
        pre = self.sess.run(self._predict, feed_dict={self.x: data})
        return pre

    def _init_varible(self, shape, name):
        """
        Initializing tensorflow variables
        :param shape: Variable shape
        :param name: Variable name
        :return: tensorflow variable
        """
        return tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=shape, stddev=self.std), name=name)


if __name__ == '__main__':
    data_dict = dh.load_mnist()
    cnn = CNN(input_dim=784, out_dim=10)
    cnn.train(data_dict=data_dict, itrs=5000)
