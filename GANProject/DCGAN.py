import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.tools import freeze_graph
import DataHandler as dh
import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
class DCGAN(object):
    def __init__(self, noise_dim=100, img_h=64, img_w=64, lr=0.0002, std=0.01):
        """
        Initializing DCGAN Networks
        :param noise_dim:the input noise dimension
        :param img_h: image height
        :param img_w: image width
        :param lr: learning rate
        :param std: standard deviation
        """
        self.noise_dim = noise_dim
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = 3
        self.lr = lr
        self.std = std
        self.d_dim = 1
        self.isTrian = tf.placeholder(dtype=tf.bool)  # Required for the bn algorithm
        self.gen_x = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, self.noise_dim])
        self.gen_out = self._init_generator(input=self.gen_x, isTrian=self.isTrian)
        self.gen_logis = self._init_discriminator(input=self.gen_out, isTrian=self.isTrian)
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_w, self.img_h, self.img_c], name="input_data")
        self.real_logis = self._init_discriminator(input=self.x, isTrian=self.isTrian, reuse=True)
        self._init_train_methods()

    def _init_discriminator(self, input, isTrian=True, reuse=False):
        """
        Initialization discriminator
        :param input:input data op
        :param isTrian: whether or not training status (bn)
        :param reuse: whether to reuse or not
        :return: Judgment op
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            # hidden layer 1   input=[None,64,64,3]
            conv1 = tf.layers.conv2d(input, 32, [3, 3], strides=(2, 2), padding="same")  # [none,32,32,32]
            bn1 = tf.layers.batch_normalization(conv1)
            active1 = tf.nn.leaky_relu(bn1)  # [none,32,32,32]
            # hidden layer 2
            conv2 = tf.layers.conv2d(active1, 64, [3, 3], strides=(2, 2), padding="same")  # [none,16,16,64]
            bn2 = tf.layers.batch_normalization(conv2)
            active2 = tf.nn.leaky_relu(bn2)  # [none,16,16,64]
            # hidden layer 3
            conv3 = tf.layers.conv2d(active2, 128, [3, 3], strides=(2, 2), padding="same")  # [none,8,8,128]
            bn3 = tf.layers.batch_normalization(conv3)
            active3 = tf.nn.leaky_relu(bn3)  # [none,8,8,128]
            # hidden layer 4
            conv4 = tf.layers.conv2d(active3, 256, [3, 3], strides=(2, 2), padding="same")  # [none,4,4,256]
            bn4 = tf.layers.batch_normalization(conv4)
            active4 = tf.nn.leaky_relu(bn4)  # [none,4,4,256]
            # out layer
            out_logis = tf.layers.conv2d(active4, 1, [4, 4], strides=(1, 1), padding='valid')  # [none,1,1,1]
        return out_logis

    def _init_generator(self, input, isTrian=True, reuse=False):
        """
        Initialization Generator
        :param input:input noise op
        :param isTrian: whether training status (BN)
        :param reuse: whether or not to reuse (reuse tensorflow variables)
        :return: Generate image op
        """
        with tf.variable_scope("generator", reuse=reuse):
            # input [none,1,1,noise_dim]
            conv1 = tf.layers.conv2d_transpose(input, 512, [4, 4], strides=[1, 1], padding="valid")  # [none,4,4,512]
            bn1 = tf.layers.batch_normalization(conv1)
            active1 = tf.nn.leaky_relu(bn1)  # [none,4,4,512]
            print(active1)
            # deconv layer2
            conv2 = tf.layers.conv2d_transpose(active1, 256, [3, 3], strides=[2, 2], padding='same')  # [none,8,8,256]
            bn2 = tf.layers.batch_normalization(conv2)
            active2 = tf.nn.leaky_relu(bn2)  # [none,8,8,256]
            # deconv layer3
            conv3 = tf.layers.conv2d_transpose(active2, 128, [3, 3], strides=[2, 2], padding="same")  # [none,16,16,128]
            bn3 = tf.layers.batch_normalization(conv3)
            active3 = tf.nn.leaky_relu(bn3)  # [none,16,16,128]
            # deconv layer4
            conv4 = tf.layers.conv2d_transpose(active3, 64, [3, 3], strides=[2, 2], padding="same")  # [none,32,32,64]
            bn4 = tf.layers.batch_normalization(conv4)
            active4 = tf.nn.leaky_relu(bn4)  # [none,32,32,64]
            # decov layer 5
            conv5 = tf.layers.conv2d_transpose(active4, 3, [3, 3], strides=(2, 2), padding="same")  # [none,64,64,3]
            out = tf.nn.tanh(conv5)
        return out

    def _init_train_methods(self):
        """
        Initial training methods: generators and discriminator loss functions, gradient descent methods, session
        :return: none
        """
        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logis, labels=tf.ones_like(self.real_logis)))
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis,labels=tf.zeros_like(self.gen_logis)))
        self.D_loss = self.D_loss_real+self.D_loss_fake
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.gen_logis,labels=tf.ones_like(self.gen_logis)))
        # Find the discriminator and generator training variables respectively.
        total_vars = tf.trainable_variables()
        d_vars = [var for var in total_vars if var.name.startswith("discriminator")]
        g_vars =[var for var in total_vars if var.name.startswith("generator")]
        self.D_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.D_loss,var_list=d_vars)
        self.G_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.G_loss,var_list=g_vars)
        # initi session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def gen_data(self, save_path="out/dcgan/test.png"):
        """
        Generate and save 25 images in 5 rows and 5 columns.
        :param save_path: The path to save the image.
        :return: numpy array to store the image
        """
        # Generate random noise
        batch_noise = np.random.normal(0,1,(25,1,1,100))
        # Incoming model generation data
        samples = self.sess.run(self.gen_out,feed_dict={self.gen_x:batch_noise,self.isTrian:True})
        # samples [25,64,64,3] float   -1-1
        # Input Conversion to 0-255 uint8 or    0-1 float
        samples = ((samples+1)/2 *255).astype(np.uint8)
        # print(samples.shape)
        fig = self.plot(samples)
        if not os.path.exists("out/dcgan/"):
            os.makedirs("out/dcgan/")
        plt.savefig(save_path,bbox_inches="tight")
        plt.close(fig)
        return samples

    def train(self, bath_size=64, itrs=20000 ):
        """
        Training Models
        :param bath_size:sample data size
        :param itrs: number of iterations
        :return: none
        """
        start_time = time.time()
        gd=None
        for i in range(itrs):
            # Read True Image
            batch_x = dh.read_img2numpy(batch_size=bath_size,img_h=64,img_w=64)
            # Generate random noise
            batch_noise = np.random.normal(0,1,(bath_size,1,1,100))
            # Training Discriminator
            _,D_loss_curr = self.sess.run([self.D_trainer,self.D_loss],
                                          feed_dict={self.x:batch_x,self.gen_x:batch_noise,self.isTrian:True})
            # Training Generators
            batch_noise = np.random.normal(0,1,(bath_size,1,1,100))
            _,G_loss_curr = self.sess.run([self.G_trainer,self.G_loss],
                                          feed_dict={self.gen_x:batch_noise,self.isTrian:True})
            if i %200 ==0:
                # Generated Data
                self.gen_data(save_path="out/dcgan/"+str(i)+".png")
                print("iters:",i," D_loss:",D_loss_curr," G_loss:",G_loss_curr)
                writer = tf.summary.FileWriter('logs/', self.sess.graph)
                self.sess.run(tf.global_variables_initializer())
                self.save()
                gd = tf.graph_util.convert_variables_to_constants(self.sess, tf.get_default_graph().as_graph_def(), ['add'])
                # print(gd)
                end_time = time.time()
                time_loss =end_time-start_time
                print("Time Consumption：",int(time_loss),"second")
                start_time = time.time()
            with tf.gfile.GFile('model/dcgan/model.pb', 'wb') as f:
                f.write(gd.SerializeToString())
        self.sess.close()

    def save(self, path="model/dcgan/"):
        """
        Save Model
        :param path: Save the model path.
        :return: none
        """
        self.saver.save(sess=self.sess,save_path=path)

    def restore(self, path='model/dcgan/'):
        """
        Recovery Model
        :param path: The path where the model is saved.
        :return:None
        """
        self.saver.restore(sess=self.sess,save_path=path)



    def plot(self, smaples):
        """
        Drawing images
        :param smaple: numpy data
        :return: Draw image
        """
        fig = plt.figure(figsize=(5,5))
        gs = gridspec.GridSpec(5,5)
        gs.update(wspace=0.05,hspace=0.05)
        for i,smaple in enumerate(smaples):
            ax = plt.subplot(gs[i])
            plt.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            plt.imshow(smaple)
        return fig


if __name__ == '__main__':
    gan = DCGAN()
    gan.train()
    dh.img2gif()





