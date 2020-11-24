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
        初始化CycleGAN对象
        :param img_h: 图像高度
        :param img_w: 图像宽度
        :param lr: 学习率
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
        # 判断真假A
        self.dis_fake_A = self._init_discriminator(input=self.fake_A, scope_name="discriminatorA", isTrain=self.isTrain)
        self.dis_real_A = self._init_discriminator(input=self.input_A, scope_name="discriminatorA",
                                                   isTrain=self.isTrain, reuse=True)
        # 判断真假B
        self.dis_fake_B = self._init_discriminator(input=self.fake_B, scope_name="discriminatorB", isTrain=self.isTrain)
        self.dis_real_B = self._init_discriminator(input=self.input_B, scope_name="discriminatorB",
                                                   isTrain=self.isTrain, reuse=True)

        # 初始化伪造数据A,B判断
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
        初始化判别器op
        :param input: 输入数据op
        :param scope_name: 判别器变量命名空间
        :param isTrain: 是否处于训练状态
        :param reuse: 是否复用内部参数
        :return: 判断结果
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
        初始化生成器op
        :param input: 输入数据op
        :param scope_name: 生成器变量命名空间
        :param isTrain: 是否处于训练状态
        :param reuse: 是否复用内部参数
        :return: 生成数据op
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
        初始化训练方法：损失函数，梯度下降方法，初始化Session
        :return: None
        """
        #构造损失函数
        self.cycle_loss = tf.reduce_mean(tf.abs(self.input_A-self.fake_rec_A))+tf.reduce_mean(tf.abs(self.input_B-self.fake_rec_B))
        self.g_loss_a2b =tf.reduce_mean(tf.squared_difference(self.dis_fake_B,1))
        self.g_loss_b2a = tf.reduce_mean(tf.squared_difference(self.dis_fake_A,1))
        self.g_loss =self.cycle_loss*10+self.g_loss_a2b+self.g_loss_b2a
        #构造判别器损失函数
        self.d_loss_a = tf.reduce_mean(tf.square(self.dis_fake_A_input))+tf.reduce_mean(tf.squared_difference(self.dis_real_A,1))
        self.d_loss_b = tf.reduce_mean(tf.square(self.dis_fake_B_input))+tf.reduce_mean(tf.squared_difference(self.dis_real_B,1))
        self.d_loss = self.d_loss_a+self.d_loss_b
        #寻找判别器与生成器的训练变量
        total_vars = tf.trainable_variables()
        d_vars = [var for var in total_vars if "discriminator" in var.name]
        g_vars = [var for var in total_vars if "generator" in var.name]
        self.D_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.d_loss,var_list=d_vars)
        self.G_trainer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.g_loss,var_list=g_vars)
        #初始化session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)


    def gen_data(self, data, a2b=True, save_path="out/CycleGAN/", name="test.png"):
        """
        生成数据模块
        :param data:需要转换的数据
        :param a2b: 是否是a到b转换
        :param save_path: 保存数据路径
        :param name: 保存图片名称
        :return: 生成数据numpy
        """

        if a2b is True:
            samples = self.sess.run(self.fake_B,feed_dict={
                self.input_A:data,self.isTrain:True})
        else:
            samples =self.sess.run(self.fake_A,feed_dict={self.input_B:data,self.isTrain:True})
        # samples [none,64,64,3] float [-1,1]
        data_B = ((samples+1)/2*255).astype(np.uint8)
        data_A = ((data+1)/2*255).astype(np.uint8)
        fig = self.plot(data_A=data_A,data_B=data_B)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+name)
        plt.close(fig)
        return data_B

    def train(self, data_path_A="data/Train_A/", data_path_B="data/faces/", batch_size=64, itrs=200000, save_time=500):
        """
        训练模型
        :param data_path_A: 数据A路径
        :param data_path_B: 数据B路径
        :param batch_size: 采样数据量
        :param itrs: 迭代训练次数
        :param save_time: 保存周期
        :return: None
        """
        start_time = time.time()
        gd = None
        self.sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter('logs/cyclegan/', self.sess.graph)
        # summary_info = self.sess.run(summary_merge_op)
        test_A = dh.read_img2numpy(batch_size=18,img_h=64,img_w=64,path=data_path_A)
        tf.summary.image("test_A", test_A)
        test_B = dh.read_img2numpy(batch_size=18,img_w=64,img_h=64,path=data_path_B)
        tf.summary.image("test_B", test_B)
        summary_merge_op = tf.summary.merge_all()
        summary_info = self.sess.run(summary_merge_op)
        writer.add_summary(summary_info)
        for i in range(itrs):

            #读取训练图片
            batch_A = dh.read_img2numpy(batch_size=batch_size,img_h=64,img_w=64,path=data_path_A)
            tf.summary.image("batch_A", batch_A)
            batch_B = dh.read_img2numpy(batch_size=batch_size,img_w=64,img_h=64,path=data_path_B)
            tf.summary.image("batch_B", batch_B)
            batch_fake_A,batch_fake_B,g_loss_curr,_=self.sess.run([self.fake_A,self.fake_B,self.g_loss,self.G_trainer],feed_dict={
                self.input_A:batch_A,self.input_B:batch_B,self.isTrain:True
            })
            tf.summary.image("batch_fake_A", batch_fake_A)
            tf.summary.image("batch_fake_B", batch_fake_B)
            summary_merge_op = tf.summary.merge_all()
            summary_info = self.sess.run(summary_merge_op)
            writer.add_summary(summary_info, i)
            #训练判别器
            d_loss_curr,_ = self.sess.run([self.d_loss,self.D_trainer],
                                          feed_dict={self.input_A:batch_A,self.input_B:batch_B,self.input_fake_A:batch_fake_A,
                                                     self.input_fake_B:batch_fake_B,self.isTrain:True})

            if i %save_time ==0:
                #生成数据
                tf.summary.scalar("D_loss",d_loss_curr)
                tf.summary.scalar("G_loss",g_loss_curr)
                summary_merge_op = tf.summary.merge_all()

                self.gen_data(data=test_A,a2b=True,save_path="out/CycleGAN/A2B/",name=str(i).zfill(6)+".png")
                self.gen_data(data=test_B,a2b=False,save_path="out/CycleGAN/B2A/",name=str(i).zfill(6)+".png")
                print("i:",i," D_loss",d_loss_curr," G_loss",g_loss_curr)
                self.save()
                gd = tf.graph_util.convert_variables_to_constants(self.sess, tf.get_default_graph().as_graph_def(), ['add'])
                # print(gd)
                end_time = time.time()
                time_loss =end_time-start_time
                print("时间消耗:",int(time_loss),"秒")
                start_time = time.time()
                summary_info = self.sess.run(summary_merge_op)
                writer.add_summary(summary_info, i)
            with tf.gfile.GFile('model/CycleGan/cycleModel.pb', 'wb') as f:
                f.write(gd.SerializeToString())
        self.sess.close()
        writer.close()


    def save(self, save_path="model/CycleGAN/"):
        """
        保存模型
        :param save_path: 保存模型路径
        :return: None
        """
        self.saver.save(sess=self.sess,save_path=save_path)

    def restore(self, save_path="model/CycleGAN/"):
        """
        恢复模型
        :param save_path: 保存模型路径
        :return: None
        """
        self.saver.restore(sess=self.sess,save_path=save_path)

    def plot(self, data_A, data_B):
        """
        绘制图形
        :param data_A:数据A
        :param data_B: 数据B
        :return: 绘制图像
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
    gan.train(data_path_A="data/train_A/",data_path_B="data/faces/")
    print(gan.restore())
