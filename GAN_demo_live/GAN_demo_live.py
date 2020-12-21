"""
TODO: docstring
"""
import datetime
import matplotlib.pyplot as pyplot
import sys
import tensorflow

class GAN:
    """
    TODO: docstring
    """
    def __init__(self):
        """
        Initialization.
        """
        mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets('MNIST_data/')

    def __call__(self):
        """
        Function call.
        """
        sess = tensorflow.Session()
        batch_size = 50
        z_dimensions = 100
        x_placeholder = tensorflow.placeholder(
            'float', shape = [None,28,28,1], name='x_placeholder')
        Gz = self.generator(batch_size, z_dimensions)
        Dx = self.discriminator(x_placeholder)
        Dg = self.discriminator(Gz, reuse=True)
        g_loss = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(
            logits=Dg, labels=tensorflow.ones_like(Dg)))
        d_loss_real = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(
            logits=Dx, labels=tensorflow.fill([batch_size, 1], 0.9)))
        d_loss_fake = tensorflow.reduce_mean(tensorflow.nn.sigmoid_cross_entropy_with_logits(
            logits=Dg, labels=tensorflow.zeros_like(Dg)))
        d_loss = d_loss_real + d_loss_fake
        tvars = tensorflow.trainable_variables()
        d_vars = [var for var in tvars if 'd_' in var.name]
        g_vars = [var for var in tvars if 'g_' in var.name]
        # train the discriminator
        with tensorflow.variable_scope(tensorflow.get_variable_scope(), reuse=False) as scope:
            d_trainer_fake = tensorflow.train.AdamOptimizer(0.0001).minimize(d_loss_fake, var_list=d_vars)
            d_trainer_real = tensorflow.train.AdamOptimizer(0.0001).minimize(d_loss_real, var_list=d_vars)
            g_trainer = tensorflow.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)
        # summary
        tensorflow.summary.scalar('Generator_loss', g_loss)
        tensorflow.summary.scalar('Discriminator_loss_real', d_loss_real)
        tensorflow.summary.scalar('Discriminator_loss_fake', d_loss_fake)
        d_real_count_ph = tensorflow.placeholder(tensorflow.float32)
        d_fake_count_ph = tensorflow.placeholder(tensorflow.float32)
        g_count_ph = tensorflow.placeholder(tensorflow.float32)
        tensorflow.summary.scalar('d_real_count', d_real_count_ph)
        tensorflow.summary.scalar('d_fake_count', d_fake_count_ph)
        tensorflow.summary.scalar('g_count', g_count_ph)
        d_on_generated = tensorflow.reduce_mean(self.discriminator(self.generator(batch_size, z_dimensions)))
        d_on_real = tensorflow.reduce_mean(self.discriminator(x_placeholder))
        tensorflow.summary.scalar('d_on_generated_eval', d_on_generated)
        tensorflow.summary.scalar('d_on_real_eval', d_on_real)
        images_for_tensorboard = self.generator(batch_size, z_dimensions)
        tensorflow.summary.image('Generated_images', images_for_tensorboard, 10)
        merged = tensorflow.summary.merge_all()
        logdir = "tensorboard/gan/"
        writer = tensorflow.summary.FileWriter(logdir, sess.graph)
        print(logdir)
        saver = tensorflow.train.Saver()
        sess.run(tensorflow.global_variables_initializer())
        gLoss = 0
        dLossFake, dLossReal = 1, 1
        d_real_count, d_fake_count, g_count = 0, 0, 0
        for i in range(50000):
            real_image_batch = mnist.train.next_batch(
                batch_size)[0].reshape([batch_size, 28, 28, 1])
            if dLossFake > 0.6:
                _, dLossReal, dLossFake, gLoss = sess.run(
                    [d_trainer_fake, d_loss_real, d_loss_fake, g_loss],
                    {x_placeholder: real_image_batch})
                d_fake_count += 1
            if gLoss > 0.5:
                _, dLossReal, dLossFake, gLoss = sess.run(
                    [g_trainer, d_loss_real, d_loss_fake, g_loss],
                    {x_placeholder: real_image_batch})
                g_count += 1
            if dLossReal > 0.45:
                _, dLossReal, dLossFake, gLoss = sess.run(
                    [d_trainer_real, d_loss_real, d_loss_fake, g_loss],
                    {x_placeholder: real_image_batch})
                d_real_count += 1
            if i % 10 == 0:
                real_image_batch = mnist.validation.next_batch(
                    batch_size)[0].reshape([batch_size, 28, 28, 1])
                summary = sess.run(
                    merged, {x_placeholder: real_image_batch, d_real_count_ph: d_real_count,
                             d_fake_count_ph: d_fake_count, g_count_ph: g_count})
                writer.add_summary(summary, i)
                d_real_count, d_fake_count, g_count = 0, 0, 0
            if i % 1000 == 0:
                images = sess.run(self.generator(3, z_dimensions))
                d_result = sess.run(self.discriminator(x_placeholder), {x_placeholder: images})
                print('TRAINING STEP', i, 'AT', datetime.datetime.now())
                for j in range(3):
                    print('Discriminator classification', d_result[j])
                    im = images[j, :, :, 0]
                    pyplot.imshow(im.reshape([28, 28]), cmap='Greys')
                    pyplot.show()
            if i % 5000 == 0:
                save_path = saver.save(sess, 'models/pretrained_gan.ckpt', global_step=i)
                print('saved to %s' % save_path)
        test_images = sess.run(self.generator(10, 100))
        test_eval = sess.run(self.discriminator(x_placeholder), {x_placeholder: test_images})
        real_images = mnist.validation.next_batch(10)[0].reshape([10, 28, 28, 1])
        real_eval = sess.run(self.discriminator(x_placeholder), {x_placeholder: real_images})
        for i in range(10):
            print(test_eval[i])
            pyplot.imshow(test_images[i, :, :, 0], cmap='Greys')
            pyplot.show()
        for i in range(10):
            print(real_eval[i])
            pyplot.imshow(real_images[i, :, :, 0], cmap='Greys')
            pyplot.show()

    def discriminator(self, x_image, reuse=False):
        """
        TODO: docstring
        """
        if (reuse):
            tensorflow.get_variable_scope().reuse_variables()
        d_w1 = tensorflow.get_variable(
            'd_w1', [5, 5, 1, 32],
            initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        d_b1 = tensorflow.get_variable(
            'd_b1', [32], initializer=tensorflow.constant_initializer(0))
        d1 = tensorflow.nn.conv2d(input=x_image, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 += d_b1
        d1 = tensorflow.nn.relu(d1)
        d1 = tensorflow.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        d_w2 = tensorflow.get_variable(
            'd_w2', [5, 5, 32, 64],
            initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        d_b2 = tensorflow.get_variable(
            'd_b2', [64], initializer=tensorflow.constant_initializer(0))
        d2 = tensorflow.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tensorflow.nn.relu(d2)
        d2 = tensorflow.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        d_w3 = tensorflow.get_variable(
            'd_w3', [7 * 7 * 64, 1024],
            initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        d_b3 = tensorflow.get_variable(
            'd_b3', [1024], initializer=tensorflow.constant_initializer(0))
        d3 = tensorflow.reshape(d2, [-1, 7 * 7 * 64])
        d3 = tensorflow.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tensorflow.nn.relu(d3)
        d_w4 = tensorflow.get_variable(
            'd_w4', [1024, 1], initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        d_b4 = tensorflow.get_variable('d_b4', [1], initializer=tensorflow.constant_initializer(0))
        d4 = tensorflow.matmul(d3, d_w4) + d_b4
        return d4

    def generator(self, batch_size, z_dim):
        """
        Takes random inputs, and mapping them down to a [1,28,28] pixel to match the MNIST data shape.
        Begin by generating a dense 14×14 set of values, and then run through a handful of filters of
        varying sizes and numbers of channels weight matrices get progressively smaller.
        """
        z = tensorflow.truncated_normal([batch_size, z_dim], mean=0, stddev=1, name='z')
        # first layer
        g_w1 = tensorflow.get_variable(
            'g_w1', [z_dim, 3136], dtype=tensorflow.float32,
            initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        g_b1 = tensorflow.get_variable(
            'g_b1', [3136], initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        g1 = tensorflow.matmul(z, g_w1) + g_b1
        g1 = tensorflow.reshape(g1, [-1, 56, 56, 1])
        g1 = tensorflow.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
        g1 = tensorflow.nn.relu(g1)
        # generate 50 features
        g_w2 = tensorflow.get_variable(
            'g_w2', [3, 3, 1, z_dim/2], dtype=tensorflow.float32,
            initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        g_b2 = tensorflow.get_variable(
            'g_b2', [z_dim/2], initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        g2 = tensorflow.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
        g2 = g2 + g_b2
        g2 = tensorflow.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
        g2 = tensorflow.nn.relu(g2)
        g2 = tensorflow.image.resize_images(g2, [56, 56])
        # generate 25 features
        g_w3 = tensorflow.get_variable(
            'g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tensorflow.float32,
            initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        g_b3 = tensorflow.get_variable(
            'g_b3', [z_dim/4], initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        g3 = tensorflow.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
        g3 = g3 + g_b3
        g3 = tensorflow.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
        g3 = tensorflow.nn.relu(g3)
        g3 = tensorflow.image.resize_images(g3, [56, 56])
        # final layer with one output channel
        g_w4 = tensorflow.get_variable(
            'g_w4', [1, 1, z_dim/4, 1], dtype=tensorflow.float32,
            initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        g_b4 = tensorflow.get_variable(
            'g_b4', [1], initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
        g4 = tensorflow.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
        g4 = g4 + g_b4
        g4 = tensorflow.sigmoid(g4)
        return g4

def main(argv):
    """
    TODO: docstring
    """
    gan = GAN
    gan()

if __name__ == '__main__':
    main(sys.argv)
