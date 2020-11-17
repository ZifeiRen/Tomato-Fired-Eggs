import tensorflow as tf
import numpy as np
import DataHandler as dh


class DeepNet(object):

    def __init__(self, input_dim=784, hidden_layer=[100, 100], out_dim=10, lr=0.001, std=0.1):
        """
        Initialized deep neural network model
        :param input_dim: input data dimension
        :param hiden_layer: the number of hidden layers.
        :param out_dim: Output dimensions
        :param lr: learning rate
        :param std: standard deviation
        """
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.out_dim = out_dim
        self.lr = lr
        self.std = std
        self._init_net()  # Initialized network structure
        self._init_train_method()  # Initial Training Methods

    def _init_variable(self, shape, name):
        """
        Initializing variables
        :param shape:variable shape
        :param name: Variable name
        :return: returns the initialized variable
        """
        return tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=shape, stddev=self.std), name=name)

    def _init_net(self):
        """
        Initialized network structure
        :return:
        """
        # Placeholders for constructing data eigenvalues and target values.
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name="input_dim")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim], name="labels")
        #  Structured Network Structure
        #  Construct the input layer
        w = self._init_variable(shape=[self.input_dim, self.hidden_layer[0]], name="w0")
        b = self._init_variable(shape=[self.hidden_layer[0]], name='b0')
        affine = tf.matmul(self.x, w) + b  # x*w+b
        hidden = tf.nn.relu(affine)
        # Constructing hidden layers
        for i in range(len(self.hidden_layer) - 1):
            with tf.name_scope("hidden_layer_" + str(i + 1)):
                w = self._init_variable(shape=[self.hidden_layer[i], self.hidden_layer[i + 1]],
                                        name="w_" + str(i + 1))
                b = self._init_variable(shape=[self.hidden_layer[i + 1]], name="b_" + str(i + 1))
                affine = tf.matmul(hidden, w) + b
                hidden = tf.nn.relu(affine)
        # Constructing the output layer
        w = self._init_variable(shape=[self.hidden_layer[-1], self.out_dim], name="w_out")
        b = self._init_variable(shape=[self.out_dim], name="b_out")
        self.out_scores = tf.matmul(hidden, w) + b

    def _init_train_method(self):
        """
        Initialized training method, loss function, gradient descent, session
        :return:
        """
        # Construct Loss Functions
        with tf.name_scope("cross_entropy"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.out_scores, labels=self.y))
        # Construct correctness calculation
        with tf.name_scope("acc"):
            self._predict = tf.argmax(self.out_scores, 1)
            correct_predict = tf.equal(tf.argmax(self.out_scores, 1), tf.argmax(self.y, 1), name="correct_predict")
            self.acc = tf.reduce_mean(tf.cast(correct_predict, tf.float32)) \
 \
                # Construct optimization methods
        with tf.name_scope("optimizer"):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # init Session
        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=1)

    def train(self, data_dict, itrs=10000, batch_size=100):
        """
        Training Network Model
        :param data_dict:data dictionary
        :param itrs: number of iterations
        :param batch_size: batch size, amount of data sampled each time
        :return:
        """
        for i in range(itrs):
            mask = np.random.choice(data_dict['train_x'].shape[0], batch_size, replace=True)
            batch_x = data_dict['train_x'][mask]
            bacth_y = data_dict['train_y'][mask]

            self.sess.run(self.train_step, feed_dict={self.x: batch_x, self.y: bacth_y})

            if i % 1000 == 0:
                temp_loss, temp_acc = self.test(data=data_dict['test_x'], labels=data_dict['test_y'])
                print("Iteration", i, "times, current losses", temp_loss, ",Current correctness", temp_acc)
                self.save()
        self.sess.close()

    def test(self, data, labels):
        """
        Test Network Model
        :param data:test data
        :param labels: The real target value of the test data.
        :return:
        """
        temp_loss, temp_acc = self.sess.run([self.loss, self.acc], feed_dict={self.x: data, self.y: labels})
        return temp_loss, temp_acc

    def predict(self, data):
        """
        Predictive data target values
        :param data: Input data
        :return: Predict target value
        """
        predict = self.sess.run(self._predict, feed_dict={self.x: data})
        return predict

    def save(self, path="model/dnn/"):
        """
        Save the training model
        :param path: Save the model path.
        :return:
        """
        self.saver.save(sess=self.sess, save_path=path)

    def restore(self, path="model/dnn/"):
        """
        Recovering a model from disk
        :param path: Restores the model path.
        :return:
        """
        self.saver.restore(sess=self.sess, save_path=path)


if __name__ == '__main__':
    data_dict = dh.load_mnist()
    deepNet = DeepNet(input_dim=784, out_dim=10)
    deepNet.restore()
    deepNet.train(data_dict=data_dict, itrs=20000)
