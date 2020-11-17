import numpy as np
import tensorflow as tf
import DataHandler as dh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


class GAN(object):

    def __init__(self, noise_dim=10, gen_hidden=[100, 100], gen_dim=784, d_hidden=[100, 100], lr=0.001, std=0.01):
        """
         Initializing GAN objects
        :param noise_dim:random_noise_dimension
        :param gen_hidden: generator hidden layer shape
        :param gen_dim: generator output dimensions
        :param d_hidden: discriminator hidden layer shape
        :param lr: learning rate
        :param std: standard deviation of weights
        """
        self.noise_dim = noise_dim
        self.gen_hidden = gen_hidden
        self.gen_dim = gen_dim
        self.d_hidden = d_hidden
        self.lr = lr
        self.std = std
        self.d_dim = 1  # Discriminator Output Dimensions
        self._init_w_g()
        self._init_w_d()
        # Construct Generator Network Structure
        self.gen_out = self._init_gen()
        self.gen_logis = self._init_dicriminator(self.gen_out)
        # Constructing Discriminator Network Structures
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.gen_dim], name="input_data")
        self.real_logis = self._init_dicriminator(self.x)
        # Initial Training Methods
        self._init_train_methods()

    def _init_w_g(self):
        """
        Initializer Weights
        :return:
        """
        self.w_g_list = []
        self.b_g_list = []
        self.w_g = []
        # Initializing input layer weights
        g_w = self._init_varible(shape=[self.noise_dim, self.gen_hidden[0]], name="gen_w0")
        g_b = self._init_varible(shape=[self.gen_hidden[0]], name='gen_b0')
        self.w_g_list.append(g_w)
        self.b_g_list.append(g_b)
        self.w_g.append(g_w)
        self.w_g.append(g_b)
        # Initialization generator hidden layer weights  [100,100]
        for i in range(len(self.gen_hidden) - 1):
            g_w = self._init_varible(shape=[self.gen_hidden[i], self.gen_hidden[i + 1]], name='gen_w' + str(i + 1))
            g_b = self._init_varible(shape=[self.gen_hidden[i + 1]], name='gen_b' + str(i + 1))
            self.w_g_list.append(g_w)
            self.b_g_list.append(g_b)
            self.w_g.append(g_w)
            self.w_g.append(g_b)
        # Initialization Generator Output Layer Weights
        g_w = self._init_varible(shape=[self.gen_hidden[-1], self.gen_dim], name="gen_w_out")
        g_b = self._init_varible(shape=[self.gen_dim], name="gen_b_out")
        self.w_g_list.append(g_w)
        self.b_g_list.append(g_b)
        self.w_g.append(g_w)
        self.w_g.append(g_b)

    def _init_w_d(self):
        """
        Initialized discriminator weights
        :return:
        """
        self.w_d_list = []
        self.b_d_list = []
        self.w_d = []
        # Initialize the discriminator input layer weights.
        d_w = self._init_varible(shape=[self.gen_dim, self.d_hidden[0]], name='d_w0')
        d_b = self._init_varible(shape=[self.d_hidden[0]], name="d_b0")
        self.w_d_list.append(d_w)
        self.b_d_list.append(d_b)
        self.w_d.append(d_w)
        self.w_d.append(d_b)
        # Initializing discriminator hidden layer weights
        for i in range(len(self.d_hidden) - 1):
            d_w = self._init_varible(shape=[self.d_hidden[i], self.d_hidden[i + 1]], name='d_w' + str(i + 1))
            d_b = self._init_varible(shape=[self.d_hidden[i + 1]], name="d_b" + str(i + 1))
            self.w_d_list.append(d_w)
            self.b_d_list.append(d_b)
            self.w_d.append(d_w)
            self.w_d.append(d_b)
        # Initialize the discriminator output layer weights.
        d_w = self._init_varible(shape=[self.d_hidden[-1], self.d_dim], name='d_w_out')
        d_b = self._init_varible(shape=[self.d_dim], name="d_b_out")
        self.w_d_list.append(d_w)
        self.b_d_list.append(d_b)
        self.w_d.append(d_w)
        self.w_d.append(d_b)

    def _init_dicriminator(self, input_op):
        """
         Initialized Discriminator Network Structure
        Network structure Example: [gen_dim,100]*[100,100]*[100,1]
        :param input_op: input op
        :return: discriminatorop
        """
        # Construct discriminator input layer
        active = tf.nn.relu(tf.matmul(input_op, self.w_d_list[0]) + self.b_d_list[0])
        # Constructing discriminator hidden layers
        for i in range(len(self.d_hidden) - 1):
            active = tf.nn.relu(tf.matmul(active, self.w_d_list[i + 1]) + self.b_d_list[i + 1])
        # Construct discriminator output layer
        out_logis = tf.matmul(active, self.w_d_list[-1]) + self.b_d_list[-1]
        return out_logis

    def _init_gen(self):
        """
        Initializer Network Structure
        Network structure e.g. [noise_dim,100]*[100,100]*[100,gen_dim]
        :return: Generate data op
        """
        self.gen_x = tf.placeholder(dtype=tf.float32, shape=[None, self.noise_dim], name="gen_x")
        # Construct generator input layer
        active = tf.nn.relu(tf.matmul(self.gen_x, self.w_g_list[0]) + self.b_g_list[0])
        # Construct Builder Hidden Layer
        for i in range(len(self.gen_hidden) - 1):
            active = tf.nn.relu(tf.matmul(active, self.w_g_list[i + 1]) + self.b_g_list[i + 1])
        # Constructing the output layer
        out_logis = tf.matmul(active, self.w_g_list[-1]) + self.b_g_list[-1]
        g_out = tf.nn.sigmoid(out_logis)
        return g_out

    def _init_varible(self, shape, name):
        """
        Initializing variables
        :param shape: Variable shape
        :param name: Variable name
        :return: tensorflow variable
        """
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=self.std), name=name)

    def train(self, data_dict, batch_size=100, itrs=100000):
        """
        Training Models
        :param data_dict:training data dictionary
        :param batch_size: batch sample size
        :param itrs: number of iterations
        :return:
        """
        for i in range(itrs):
            mask = np.random.choice(data_dict['train_x'].shape[0], batch_size, replace=True)
            batch_x = data_dict['train_x'][mask]
            batch_noise = self.sample_Z(m=batch_size, n=self.noise_dim)
            _, D_loss_curr = self.sess.run([self.D_trainer, self.D_loss],
                                           feed_dict={self.x: batch_x, self.gen_x: batch_noise})

            _, G_loss_curr = self.sess.run([self.G_trainer, self.G_loss], feed_dict={self.gen_x: batch_noise})
            if i % 1000 == 0:
                self.gen_data(save_path="out/" + str(i) + ".png")
                print("i:,", i, " D_loss:", D_loss_curr, " G_loss:", G_loss_curr)
                self.save()
        self.sess.close()

    def _init_train_methods(self):
        """
        Initialization discriminator and generator loss functions, training methods, sessions.
        :return:
        """
        # Initializing the discriminator loss function
        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logis, labels=tf.ones_like(self.real_logis)))
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis, labels=tf.zeros_like(self.gen_logis)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        # Initialization generator loss values
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis, labels=tf.ones_like(self.gen_logis)))
        # Constructive Training Methods
        self.D_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.w_d)
        self.G_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=self.w_g)
        # inits Session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def gen_data(self, save_path="out/test.png"):
        """
        Generate and save images
        :param save_path:save_path
        :return:
        """
        batch_noise = self.sample_Z(9, self.noise_dim)
        sample = self.sess.run(self.gen_out, feed_dict={self.gen_x: batch_noise})
        fig = self.plot(sample)
        if not os.path.exists("out/"):
            os.makedirs("out/")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return sample

    def save(self, path="model/gan/"):
        """
        Save Model
        :param path: Save the model path.
        :return:
        """
        self.saver.save(self.sess, save_path=path)

    def restore(self, path="model/gan/"):
        """
        Recovery Model
        :param path: The path saved by the model.
        :return:
        """
        self.saver.restore(sess=self.sess, save_path=path)

    def sample_Z(self, m, n):
        """
        Generate random noise
        :param m: Generate data volume
        :param n: random noise dimension
        :return: numpy array
        """
        return np.random.uniform(-1., 1., size=[m, n])

    def plot(self, samples):
        """
        Drawing images
        :param sample: Draw data (numpy)
        :return:
        """
        fig = plt.figure(figsize=(3, 3))
        gs = gridspec.GridSpec(3, 3)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        return fig


if __name__ == '__main__':
    # Read Data
    data = dh.load_mnist()
    # Initializing GAN objects
    gan = GAN(noise_dim=10, gen_dim=784)
    gan.train(data_dict=data)
