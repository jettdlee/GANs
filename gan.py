#######################################################################################################################
#   GANS Network
#   Created by Jet-Tsyn Lee, 02/07/18
#   Last Update 10/07/18 v0.9
#######################################################################################################################

import os
import tensorflow as tf
import numpy as np
from utils import *
import time
slim = tf.contrib.slim



class Gan:


    def __init__(self, batch_size, start_time, srt,data_dest, result_dest, ckp_dir):
        # size of resulting image to process
        self.height, self.width, self.channel = 128, 128, 3
        self.batch_size = batch_size
        self.time = start_time
        self.srt = srt

        # Directories and names
        self.dataset_dir = data_dest            # Folder containing images to train on
        self.output_dir = result_dest           # Folder to save generated images
        self.ckp_dir = ckp_dir                  # Main folder storing checkpoints
        self.new_ckp = os.path.join(self.ckp_dir,"latest")  # folder to load checkpoints


        # Network parameters
        self.kernel_size=[5, 5]
        self.strides=[2, 2]
        self.epsilon=1e-5
        self.decay = 0.9
        self.stddev=0.02
        self.z_dim = 100
        self.learning_rate = 2e-4



        # IMPORT IMAGES
        self.images = []
        for data in os.listdir(self.dataset_dir):
            self.images.append(os.path.join(self.dataset_dir,data))
        self.total_images = len(self.images)
        all_images = tf.convert_to_tensor(self.images, dtype = tf.string)
        images_queue = tf.train.slice_input_producer([all_images])
        self.content = tf.read_file(images_queue[0])



    def get_batch(self):

        image = tf.image.decode_jpeg(self.content, channels = self.channel)

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta = 0.1)
        image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)

        size = [self.height, self.width, self.channel]
        image = tf.image.resize_images(image, size[:-1])
        image.set_shape(size)

        image = tf.cast(image, tf.float32)
        image = image / 255.0

        img_batch = tf.train.shuffle_batch([image], batch_size=self.batch_size, num_threads=4, capacity=200+3*self.batch_size, min_after_dequeue = 200)

        return img_batch




    ###################################################################################################
    # GENERATOR
    def generator(self, input, random_dim, is_train, reuse=False):

        channel_arr = [512, 256, 128, 64, 32]
        size = 4
        output_dim = self.channel  # RGB image


        with tf.variable_scope('Generator') as scope:
            if reuse:
                scope.reuse_variables()

            weight = tf.get_variable('w1', shape=[random_dim, size * size * channel_arr[0]], dtype=tf.float32,initializer=norm_init(stddev=0.02))
            bias = tf.get_variable('b1', shape=[channel_arr[0] * size * size], dtype=tf.float32,initializer=cons_init(0.0))
            flat_conv = tf.add(tf.matmul(input, weight), bias, name='flat_conv1')
            active_lay = flat_conv # initialize variable to aviod error

            for i in range(len(channel_arr)):
                lbl = str(i+1)

                # Reshape initial layer
                if i == 0:
                    conv_lay = tf.reshape(flat_conv, shape=[-1, size, size, channel_arr[i]], name='convolution_'+lbl)
                else:
                    conv_lay = conv2d_tran(active_lay, channel_arr[i], kernal_size=self.kernel_size, strides=self.strides, stddev=self.stddev,name="convolution_"+lbl)

                bn_lay = batch_norm(conv_lay, epsilon=self.epsilon, decay=self.decay, name="batch_norm_"+lbl, is_train=is_train, scale=False)
                active_lay = tf.nn.relu(bn_lay, name='act'+lbl)

            conv_lay = conv2d_tran(active_lay, output_dim, kernal_size=self.kernel_size, strides=self.strides, stddev=self.stddev,name="conv6")
            # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
            active_lay = tf.nn.tanh(conv_lay, name='activation_'+str(len(channel_arr)+1))

            return active_lay



    ###################################################################################################
    # DISCRIMINATOR
    def discriminator(self, input, is_train, reuse=False):

        channel_arr = [64, 128, 256, 512]

        with tf.variable_scope('Discriminator') as scope:

            if reuse:
                scope.reuse_variables()

            # Loop each layer
            active_lay = input
            for i in range(len(channel_arr)):
                lbl = str(i+1)  # Label
                conv_lay = conv2d(active_lay,channel_arr[i],kernal_size=self.kernel_size,strides=self.strides,stddev=self.stddev,name='convolution_'+lbl)
                bn_lay = batch_norm(conv_lay, epsilon=self.epsilon, decay=self.decay, name="batch_norm_"+lbl, is_train=is_train, scale=False)

                # use conv layer for first loop
                if i == 0:
                    active_lay = lrelu(conv_lay, name='activation_'+lbl)
                else:
                    active_lay = lrelu(bn_lay, name='activation_'+lbl)


            dim = int(np.prod(active_lay.get_shape()[1:]))

            fc_lay = tf.reshape(active_lay, shape=[-1, dim], name='fc1')
            weight = tf.get_variable('weight', shape=[fc_lay.shape[-1], 1], dtype=tf.float32,initializer=norm_init(stddev=self.stddev))
            bias = tf.get_variable('bias', shape=[1], dtype=tf.float32,initializer=cons_init(0.0))

            logits = tf.add(tf.matmul(fc_lay, weight), bias, name='logits')
            sig = tf.nn.sigmoid(logits)


            return logits


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CHECKPOINTS AND IMAGES



    # LOAD CHECKPOINT
    def ckp_load(self):

        ckpt = tf.train.get_checkpoint_state(self.new_ckp)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.new_ckp, ckpt_name))
            print("Checkpoint Loaded")
        else:
            print("Checkpoint load failed")


    # SAVE CHECKPOINT
    def ckp_save(self, step):
        # checkpoint wil save in two locations,
        # in the current date to store and act as historic
        # latest folder for code to load in future runs

        # Create folder
        ckp_fold = create_path(self.ckp_dir)    # Main checkpoint folder
        run_fold = create_path(os.path.join(ckp_fold, self.time)) # Date folder
        new_fold = create_path(self.new_ckp)    # newest checkpoint

        # Due to space limitations, will delete all previous checkpoints in the current run
        for file in os.listdir(run_fold):
            os.remove(run_fold+"/"+file)
        for file in os.listdir(new_fold):
            os.remove(new_fold+"/"+file)

        loc_arr = [run_fold, new_fold]
        for i in range(len(loc_arr)):
            # Save checkpoint
            save_loc = os.path.join(loc_arr[i],"model")
            self.saver.save(self.sess,save_loc,global_step=step)

            # Create log
            file = open(os.path.join(loc_arr[i],"log.txt"),"w")
            file.write("Run started: "+self.time)
            file.write("\nIterations complete: %d"%(step))
            file.write("\nCurrent runtime: "+timer(self.srt,time.time()))
            file.close()


    # SAVE IMAGE
    def img_save(self, input, step):
        # save images
        save1 = create_path(self.output_dir)                        # main inage folder
        save2 = create_path(os.path.join(save1, self.time))         # date folder
        save_path = create_path(os.path.join(save2,"Iteration "+str(step))) # iteration folder
        img_name = start_time + "_iter" + str(step)# + '.jpg'

        #save_path = os.path.join(self.output_dir,img_name)
        #save_images(input, [3,3] ,save_path)
        save_images(input, save_path, img_name)






    ###################################################################################################
    # TRAINING
    def train(self, iterations, save_iter=0, ckp_iter=0, load_prev=False):

        # TENSORFLOW VARIABLES
        image_batch = self.get_batch()
        batch_num = int(self.total_images / self.batch_size)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Varaibles
        with tf.variable_scope('input'):
            #real and fake image placholders
            x_image = tf.placeholder(tf.float32, shape = [None, self.height, self.width, self.channel], name='real_image')
            z_noise = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='rand_input')
            y_input = tf.placeholder(tf.float32, shape = [None, self.height, self.width, self.channel], name='adapt_image')
            train_phase = tf.placeholder(tf.bool, name='is_train')

            output_c_dim = 3
            self.real_data = tf.placeholder(tf.float32,[self.batch_size, self.height, self.width,self.channel + output_c_dim],name='real_A_and_B_images')
            self.real_B = self.real_data[:, :, :, :self.channel]
            self.real_A = self.real_data[:, :, :, self.channel:self.channel + output_c_dim]




        self.fake_B = self.generator(self.real_A)


        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)

        G_z = self.discriminator(x_image, train_phase)   # test genereator against real image
        D_g = self.discriminator(D_x, train_phase, reuse=True)   # test generator against fake image


        # Create network
        # Generate image from generator
        D_x = self.generator(z_noise, self.z_dim, train_phase)

        # apply to discriminator to test
        G_z = self.discriminator(x_image, train_phase)   # test genereator against real image
        D_g = self.discriminator(D_x, train_phase, reuse=True)   # test generator against fake image


        # ERROR
        # Calculate error of the networks
        d_loss = tf.reduce_mean(D_g) - tf.reduce_mean(G_z)  # This optimizes the discriminator.
        g_loss = -tf.reduce_mean(D_g)  # This optimizes the generator.

        #define our optimizers to update weights
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'Generator' in var.name]

        '''


        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.sampler(self.real_A)


        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))+self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)


        '''



















        # Update network, use RMS gradient decent
        trainer_d = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(d_loss, var_list=self.d_vars)
        trainer_g = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(g_loss, var_list=self.g_vars)


        # clip discriminator weights
        d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_vars]



        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # START SESSION & LOAD CHECKPOINT
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        load_discriminator = True
        if load_discriminator:
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.Saver(self.g_vars)

        # Load Checkpoint
        if load_prev:
            self.ckp_load()



        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        print('\n\ntotal training sample num:%d' % self.total_images)
        print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (self.batch_size, batch_num, iterations))
        print('start training...\n\n')



        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # BEGIN LOOP
        for i in range(iterations+1):
            print("Running epoch {}/{}...".format(i, iterations))

            # Training
            for j in range(batch_num):
                train_img = self.sess.run(image_batch)

                # TRAIN DISCRIMINATOR
                z_batch = rand_noise(self.batch_size, self.z_dim)
                self.sess.run(d_clip)
                _, dLoss = self.sess.run([trainer_d, d_loss], feed_dict={z_noise: z_batch, x_image: train_img, train_phase: True})

                # Update the generator
                z_batch = rand_noise(self.batch_size, self.z_dim)
                _, gLoss = self.sess.run([trainer_g, g_loss], feed_dict={z_noise: z_batch, train_phase: True})








            # SAVE CHECKPOINT
            if ckp_iter != 0 and i%ckp_iter == 0:
                print("Saving checkpoint")
                self.ckp_save(i)

            # SAVE IMAGES
            if save_iter != 0 and i%save_iter == 0:
                print("Saving image")
                sample_noise = rand_noise(self.batch_size, self.z_dim)
                imgtest = self.sess.run(D_x, feed_dict={z_noise: sample_noise, train_phase: False})
                self.img_save(imgtest,i)

            # print loss after 10 iterations
            if i%10 == 0:
                print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))


        coord.request_stop()
        coord.join(threads)









    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




if __name__ == "__main__":

    start_time = time.strftime("%d%b%Y%H%M%S", time.localtime())
    srt = time.time()

    iterations = 5  # no of training iterations
    save_iter = 1   # no of iterations before saving image, 0 = dont save
    ckp_iter = 0    # no of iteration before saving checkpoint, 0 = dont save

    batch_size = 9
    dataset_dir = "./dataset"
    result_dir = "./results"
    ckp_dir = "./checkpoint"
    load_prev = True

    gan = Gan(batch_size, start_time, srt, dataset_dir, result_dir, ckp_dir)

    gan.train(iterations, save_iter, ckp_iter, load_prev)

    print("\n\n---COMPLETE---\nRuntime:",timer(srt,time.time()))
