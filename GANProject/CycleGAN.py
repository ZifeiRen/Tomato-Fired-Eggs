import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import DataHandler as dh


class CycleGAN(object):

    def __init__(self, img_h=64, img_w=64, lr=0.0004):
        """
        Initializing checkpoints objects
        :param img_h: image height
        :param img_w: image width
        :param lr: learning rate
        """
        self.img_h = img_h
        self.img_w = img_w
        self.lr = lr
        self.d_dim = 1
        with tf.name_scope("isTrain"):
            self.isTrain = tf.placeholder(dtype=tf.bool)
        self.img_c = 3
        # A--->B--->A'
        with tf.name_scope("input_A"):
            self.input_A = tf.placeholder(dtype=tf.float32, shape=[None, self.img_w, self.img_h, self.img_c])

        self.fake_B = self._init_generator(input=self.input_A, scope_name="generatorA2B", isTrain=self.isTrain)

        self.fake_rec_A = self._init_generator(input=self.fake_B, scope_name="generatorB2A", isTrain=self.isTrain)

        # B--->A---->B'
        with tf.name_scope("input_B"):
            self.input_B = tf.placeholder(dtype=tf.float32, shape=[None, self.img_w, self.img_h, self.img_c])
        self.fake_A = self._init_generator(input=self.input_B, scope_name="generatorB2A", isTrain=self.isTrain,
                                           reuse=True)
        self.fake_rec_B = self._init_generator(input=self.fake_A, scope_name="generatorA2B", isTrain=self.isTrain,
                                               reuse=True)
        #Determining True/False A
        self.dis_fake_A = self._init_discriminator(input=self.fake_A, scope_name="discriminatorA", isTrain=self.isTrain)
        self.dis_real_A = self._init_discriminator(input=self.input_A, scope_name="discriminatorA",
                                                   isTrain=self.isTrain, reuse=True)
        # Determining True/False B
        self.dis_fake_B = self._init_discriminator(input=self.fake_B, scope_name="discriminatorB", isTrain=self.isTrain)
        self.dis_real_B = self._init_discriminator(input=self.input_B, scope_name="discriminatorB",
                                                   isTrain=self.isTrain, reuse=True)

        # Initialize forged data A, B judgments
        with tf.name_scope("input_fake_A"):
            self.input_fake_A = tf.placeholder(dtype=tf.float32, shape=[None, self.img_w, self.img_h, self.img_c],
                                               name="fake_A_data")
        with tf.name_scope("input_fake_B"):
            self.input_fake_B = tf.placeholder(dtype=tf.float32, shape=[None, self.img_w, self.img_h, self.img_c],
                                               name="fake_B_data")
        self.dis_fake_A_input = self._init_discriminator(self.input_fake_A, scope_name="discriminatorA",
                                                         isTrain=self.isTrain, reuse=True)
        self.dis_fake_B_input = self._init_discriminator(self.input_fake_B, scope_name="discriminatorB",
                                                         isTrain=self.isTrain, reuse=True)
        self._init_train_methods()



    def _init_discriminator(self, input, scope_name="discriminator", isTrain=True, reuse=False):
        """
        Initialization discriminator op
        :param input: input data op
        :param scope_name: Discriminator variable namespace
        :param isTrain: Is in training or not.
        :param reuse: whether or not to reuse internal parameters
        :return: Judgment result
        """
        with tf.variable_scope(scope_name,reuse=reuse):
            # input [none,64,64,3]
            conv1 = tf.layers.conv2d(input,32,[4,4],strides=(2,2),padding="same") #[none,32,32,32]
            bn1 = tf.layers.batch_normalization(conv1,training=isTrain)
            active1 = tf.nn.leaky_relu(bn1) #[none,32,32,32]
            #layer 2
            conv2 = tf.layers.conv2d(active1,64,[4,4],strides=(2,2),padding="same") #[none,16,16,64]
            bn2 = tf.layers.batch_normalization(conv2,training=isTrain)
            active2 = tf.nn.leaky_relu(bn2) #[none,16,16,64]
            # layer 3
            conv3 = tf.layers.conv2d(active2,128,[4,4],strides=(2,2),padding='same') #[none,8,8,128]
            bn3 = tf.layers.batch_normalization(conv3,training=isTrain)
            active3 = tf.nn.leaky_relu(bn3) #[none,8,8,128]
            #layer 4
            conv4 = tf.layers.conv2d(active3,256,[4,4],strides=(2,2),padding="same") #[none,4,4,256]
            bn4 = tf.layers.batch_normalization(conv4,training=isTrain)
            active4 = tf.nn.leaky_relu(bn4) #[none,4,4,256]
            # out layer
            out_logis = tf.layers.conv2d(active4,1,[4,4],strides=(1,1),padding="valid") #[none,1,1,1]
        return out_logis


    def _init_generator(self, input, scope_name='generator', isTrain=True, reuse=False):
        """
        Initializer op
        :param input: input data op
        :param scope_name: generator variable namespace
        :param isTrain: Is in training or not.
        :param reuse: whether or not to reuse internal parameters
        :return: Generate data op
        """
        with tf.variable_scope(scope_name,reuse=reuse):
            #input[none,64,64,3]

            conv1 = tf.layers.conv2d(input,64,[4,4],strides=(2,2),padding="same")# [none,32,32,64]
            bn1 = tf.layers.batch_normalization(conv1,training=isTrain)
            active1 = tf.nn.leaky_relu(bn1) #[none,32,32,64]
            #layer 2
            conv2 = tf.layers.conv2d(active1,128,[4,4],strides=(2,2),padding="same") #[none,16,16,128]
            bn2 = tf.layers.batch_normalization(conv2,training=isTrain)
            active2 = tf.nn.leaky_relu(bn2) #[none,16,15,128]
            # layer3
            conv3 = tf.layers.conv2d(active2,256,[4,4],strides=(2,2),padding="same") #[none,8,8,256]
            bn3 = tf.layers.batch_normalization(conv3,training=isTrain)
            active3 = tf.nn.leaky_relu(bn3) #[none,8,8,256]
            # deconv layer 1
            de_conv1 = tf.layers.conv2d_transpose(active3,128,[4,4],strides=(2,2),padding="same") #[none,16,16,128]
            de_bn1=tf.layers.batch_normalization(de_conv1,training=isTrain)
            de_active1 = tf.nn.leaky_relu(de_bn1) #[none,16,16,128]
            # deconv layer 2
            de_conv2 = tf.layers.conv2d_transpose(de_active1,64,[4,4],strides=(2,2),padding="same") #[none,32,32,64]
            de_bn2 = tf.layers.batch_normalization(de_conv2,training=isTrain)
            de_active2 = tf.nn.leaky_relu(de_bn2) #[none,32,32,64]
            # deconv layer 3
            de_conv3 = tf.layers.conv2d_transpose(de_active2,3,[4,4],strides=(2,2),padding="same") #[none,64,64,3]
            out = tf.nn.tanh(de_conv3)

        return out

    def _init_train_methods(self):
        """
        Initialization training methods: loss function, gradient descent method, initialization session
        :return: None
        """
        #Construct Loss Functions
        self.cycle_loss = tf.reduce_mean(tf.abs(self.input_A-self.fake_rec_A))+tf.reduce_mean(tf.abs(self.input_B-self.fake_rec_B))
        self.g_loss_a2b =tf.reduce_mean(tf.squared_difference(self.dis_fake_B,1))
        self.g_loss_b2a = tf.reduce_mean(tf.squared_difference(self.dis_fake_A,1))
        self.g_loss =self.cycle_loss*10+self.g_loss_a2b+self.g_loss_b2a
        #Construct discriminator loss function
        self.d_loss_a = tf.reduce_mean(tf.square(self.dis_fake_A_input))+tf.reduce_mean(tf.squared_difference(self.dis_real_A,1))
        self.d_loss_b = tf.reduce_mean(tf.square(self.dis_fake_B_input))+tf.reduce_mean(tf.squared_difference(self.dis_real_B,1))
        self.d_loss = self.d_loss_a+self.d_loss_b
        #Finding training variables for discriminators and generators
        total_vars = tf.trainable_variables()
        d_vars = [var for var in total_vars if "discriminator" in var.name]
        g_vars = [var for var in total_vars if "generator" in var.name]
        # Here, using the Adam optimizer, you can also use random gradient descent.
        self.D_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.d_loss,var_list=d_vars)
        self.G_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.g_loss,var_list=g_vars)
        #Initialization session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)


    def gen_data(self, data, a2b=True, save_path="out/checkpoints/", name="test.png"):
        """
        Generating Data Modules
        :param data: the data to be converted
        :param a2b: is it an a to b conversion?
        :param save_path: Save data path
        :param name: Save image name
        :return: Generate data numpy
        """

        if a2b is True:
            samples = self.sess.run(self.fake_B,feed_dict={
                self.input_A:data,self.isTrain:True})

        else:
            samples =self.sess.run(self.fake_A,feed_dict={self.input_B:data,self.isTrain:True})

        data_B = ((samples+1)/2*255).astype(np.uint8)
        data_A = ((data+1)/2*255).astype(np.uint8)
        fig = self.plot(data_A=data_A,data_B=data_B)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+name)
        plt.close(fig)
        return data_B

    def train(self, data_path_A="data/cats/", data_path_B="data/faces/", batch_size=64, itrs=25000, save_time=500):
        """
        Training Models
        :param data_path_A: path of data A
        :param data_path_B: Data B path
        :param batch_size: the amount of data to sample.
        :param itrs: number of training iterations
        :param save_time: Save the period.
        :return: None
        """
        start_time = time.time()
        test_A = dh.read_img2numpy(batch_size=18, img_h=64, img_w=64, path=data_path_A)
        test_B = dh.read_img2numpy(batch_size=18, img_w=64, img_h=64, path=data_path_B)
        for i in range(itrs):
            # Read training images
            batch_A = dh.read_img2numpy(batch_size=batch_size, img_h=64, img_w=64, path=data_path_A)
            batch_B = dh.read_img2numpy(batch_size=batch_size, img_w=64, img_h=64, path=data_path_B)
            batch_fake_A, batch_fake_B, g_loss_curr, _ = self.sess.run(
                [self.fake_A, self.fake_B, self.g_loss, self.G_trainer], feed_dict={
                    self.input_A: batch_A, self.input_B: batch_B, self.isTrain: True
                })
            # Training Discriminator
            d_loss_curr, _ = self.sess.run([self.d_loss, self.D_trainer],
                                           feed_dict={self.input_A: batch_A, self.input_B: batch_B,
                                                      self.input_fake_A: batch_fake_A,
                                                      self.input_fake_B: batch_fake_B, self.isTrain: True})
            if i % save_time == 0:
                # Generated Data
                self.gen_data(data=test_A, a2b=True, save_path="out/CycleGAN/cats2anime/", name=str(i).zfill(6) + ".png")
                self.gen_data(data=test_B, a2b=False, save_path="out/CycleGAN/anime2cats/", name=str(i).zfill(6) + ".png")
                print("i:", i, " D_loss", d_loss_curr, " G_loss", g_loss_curr)
                self.save()
                end_time = time.time()
                time_loss = end_time - start_time
                print("Time Consumption:", int(time_loss), "Seconds")
                start_time = time.time()
        self.sess.close()


    def save(self, save_path="model/catAndanime/"):
        """
        Save Model
        :param save_path: Save the model path.
        :return: None
        """
        self.saver.save(sess=self.sess,save_path=save_path)

    def restore(self, save_path="model/catAndanime/"):
        """
        Recovery Model
        :param save_path: Save the model path.
        :return: None
        """
        self.saver.restore(sess=self.sess,save_path=save_path)

    def plot(self, data_A, data_B):
        """
        Draw Graphics
        :param data_A:data_A
        :param data_B: Data B
        :return: Draw image
        """
        fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(6,6)
        gs.update(wspace=0.05,hspace=0.05)
        for i in range(data_A.shape[0]):
            ax = plt.subplot(gs[i*2])
            plt.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            plt.imshow(data_A[i])

            ax = plt.subplot(gs[i * 2+1])
            plt.axis("off")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            plt.imshow(data_B[i])
        return fig

if __name__ == '__main__':
    gan = CycleGAN()
    gan.train()
