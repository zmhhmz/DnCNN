import os
import numpy as np
import skimage.io as skio
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import tensorflow.contrib.slim as slim
from skimage.measure import compare_psnr, compare_ssim
from skimage.transform import resize
import glob

# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS

# 模式：训练、测试
tf.app.flags.DEFINE_string('mode', 'train',
                           'train or test or iniTest or testAll.')
# 学习率
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                          'learning_rate')

tf.app.flags.DEFINE_float('sigma', 25.0,
                          'noise level')

tf.app.flags.DEFINE_integer('epoch',2,
                            'epoch')

tf.app.flags.DEFINE_integer('level', 17,
                            'level number')

tf.app.flags.DEFINE_string('train_data', 'data/Train400/',
                           'Directory of training data.')

tf.app.flags.DEFINE_string('test_data', 'data/Test/Set68/',
                           'Directory of testing data.')

tf.app.flags.DEFINE_string('train_dir', 'temp/tf_test_true1/',
                           'Directory to keep training outputs.')

tf.app.flags.DEFINE_string('test_dir', 'TestResult/Set68_tf40epo_sigma25/25/',    
                           'Directory to keep eval outputs.')

tf.app.flags.DEFINE_integer('image_channel', 1,
                            'Image channel.')

tf.app.flags.DEFINE_integer('net_channel', 64,
                            'Network channel.')

tf.app.flags.DEFINE_integer('patch_size', 40,
                            'training patch size.')

tf.app.flags.DEFINE_integer('image_size', 256,
                            'size of test images')

tf.app.flags.DEFINE_integer('stride', 10,
                            'training patch stride.')

tf.app.flags.DEFINE_integer('batch_size', 16,
                            "Batch size.")
# GPU设备数量（0代表CPU）
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus. (0 or 1)')


def train():
    random.seed(0)

    x = tf.placeholder(tf.float32, shape=(
        None, FLAGS.patch_size, FLAGS.patch_size, FLAGS.image_channel))
    noisy_x = tf.placeholder(tf.float32, shape=(
        None, FLAGS.patch_size, FLAGS.patch_size, FLAGS.image_channel))
    res = DnCNN(noisy_x)
    x_ = noisy_x - res

    loss = slim.losses.mean_squared_error(x_, x)

    lr_ = FLAGS.learning_rate
    lr = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = slim.learning.create_train_op(loss, optimizer)

    saver = tf.train.Saver(max_to_keep=10)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=True)
    save_path = FLAGS.train_dir

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    epoch = int(FLAGS.epoch)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if tf.train.get_checkpoint_state(save_path):
            ckpt = tf.train.latest_checkpoint(save_path)
            saver.restore(sess, ckpt)
            start_point = int(ckpt.split('-')[-1])
            print("Load success")
        else:
            print("re-training")
            start_point = 0

        for j in range(start_point, epoch):
            if j+1 > (4*epoch/5):
                lr_ = FLAGS.learning_rate*0.1

            All_data = datagenerator(FLAGS.train_data)
            np.random.shuffle(All_data)
            num_data = All_data.shape[0]
            for iter in range(num_data//FLAGS.batch_size):
                batch_x = Batch_data(All_data, FLAGS.batch_size, iter)/255.0
                batch_noisy_x = batch_x + \
                    np.random.normal(0, FLAGS.sigma/255.0, batch_x.shape)

                _, lossvalue,X_ = sess.run([train_op, loss,x_], feed_dict={
                                        x: batch_x, noisy_x: batch_noisy_x, lr: lr_})
                if (iter+1) % 10 ==0:
                    print('epoch:%s__batch:%s/%s__loss:%s' %
                          (j+1, iter+1,num_data//FLAGS.batch_size, lossvalue))
                
                    toshow = np.hstack((batch_x[0],batch_noisy_x[0],X_[0]))
                    imshow(toshow)

            model_name = 'model-epoch'
            save_path_full = save_path + model_name
            saver.save(sess, save_path_full, global_step=j+1)

            ckpt = tf.train.latest_checkpoint(save_path)
            saver.restore(sess, ckpt)

def test():
    noisy_x = tf.placeholder(tf.float32, shape=(
        1, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channel))
    res = DnCNN(noisy_x)
    x_ = noisy_x - res
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=5)
    save_path = FLAGS.train_dir
    if not os.path.exists(FLAGS.test_dir):
        os.makedirs(FLAGS.test_dir)
    list_psnr = []
    list_ssim = []
    with tf.Session(config=config) as sess:
        ckpt = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, ckpt)
        files = os.listdir(FLAGS.test_data)
        files.sort()
        for file in files:
            np.random.seed(seed=0)
            x = np.array(skio.imread(FLAGS.test_data+file), dtype=np.float32)/255.0
            x = resize(x,(FLAGS.image_size,FLAGS.image_size))
            x = x.astype(np.float32)
            y = x + np.random.normal(0, FLAGS.sigma/255.0, x.shape)
            y = y.astype(np.float32)
            y = np.expand_dims(np.expand_dims(y,0),3)
            X_ = sess.run([x_],feed_dict={noisy_x:y})
            
            X_ = X_[0].squeeze()
            
            toshow = np.hstack((x,y.squeeze(),X_))
            toshow = np.maximum(toshow, 0)
            toshow = np.minimum(toshow, 1)
            imshow(toshow)
            skio.imsave(FLAGS.test_dir+file, toshow)
            psnr_x_ = compare_psnr(x, X_)
            ssim_x_ = compare_ssim(x, X_)#,multichannel=True)
            list_psnr.append(psnr_x_)
            list_ssim.append(ssim_x_)
            print(file+' done! PSNR: %s, SSIM: %s'%(psnr_x_,ssim_x_))
    print('==================================')
    print('Mean PSNR: %s. Mean SSIM: %s'%(np.mean(list_psnr),np.mean(list_ssim)))


def DnCNN(X):  
    with slim.arg_scope([slim.conv2d], padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01), activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(0.0005), normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': True, 'decay': 0.95}):
        X = slim.conv2d(X, FLAGS.net_channel, [3, 3],normalizer_fn=None,biases_initializer=tf.constant_initializer(0.0), scope='conv1')
        X = slim.repeat(X, FLAGS.level-2, slim.conv2d,
                        FLAGS.net_channel, [3, 3], scope='conv2')
        X = slim.conv2d(X, FLAGS.image_channel, [3, 3],biases_initializer=None, normalizer_fn=None, activation_fn=None, scope='conv%s' % (FLAGS.level))
    return X
    

def imshow(X):
    X = np.maximum(X, 0)
    X = np.minimum(X, 1)
    plt.imshow(X.squeeze(),cmap='gray')
    plt.axis('off')
    plt.show()
    

def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    scales = [1, 0.9, 0.8, 0.7]
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-FLAGS.patch_size+1, FLAGS.stride):
            for j in range(0, w_scaled-FLAGS.patch_size+1, FLAGS.stride):
                x = img_scaled[i:i+FLAGS.patch_size, j:j+FLAGS.patch_size]
                x_aug = data_aug(x, mode=np.random.randint(0, 8))
                patches.append(x_aug)
    return patches


def datagenerator(data_dir='data/Train400', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:    
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data)-len(data)//FLAGS.batch_size*FLAGS.batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data


def Batch_data(data, batch_size, iter):
    x = data[iter*FLAGS.batch_size:(iter+1)*FLAGS.batch_size]
    return x


if __name__ == '__main__':
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'test':  
            test()
        elif FLAGS.mode == 'train':
            train()
