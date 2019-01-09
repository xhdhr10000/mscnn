# -*- coding:utf-8 -*-
"""
@Function: MSCNN crowd counting model evaluation
@Source: Multi-scale Convolution Neural Networks for Crowd Counting
         https://arxiv.org/abs/1702.02359
@Data set: https://pan.baidu.com/s/12EqB1XDyFBB0kyinMA7Pqw 密码: sags  --> Have some problems

@Author: Ling Bao
@Code verification: Ling Bao
@说明：
    学习率：1e-4
    平均loss : 14.

@Data: Sep. 11, 2017
@Version: 0.1
"""

# 系统库
import os
import sys
import time
import numpy as np
import cv2

# 机器学习库
from tensorflow.python.platform import gfile
import tensorflow as tf

# 项目库
import mscnn

# 参数设置
eval_dir = 'eval'
data_test_gt = 'Data_original/Data_gt/train_gt/'
data_test_im = 'Data_original/Data_im/train_im/'
data_test_index = 'Data_original/dir_name.txt'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', eval_dir, """日志目录""")
tf.app.flags.DEFINE_string('data_test_gt', data_test_gt, """测试集集标签""")
tf.app.flags.DEFINE_string('data_test_im', data_test_im, """测试集图片""")
tf.app.flags.DEFINE_string('data_test_index', data_test_index, """测试集图片""")


def evaluate():
    """
    在ShanghaiTech测试集上对mscnn模型评价
    :return:
    """
    # 构建图模型
    images = tf.placeholder("float")
    labels = tf.placeholder("float")
    predict_op = mscnn.inference_bn(images)
    loss_op = mscnn.loss(predict_op, labels)

    # 载入模型参数
    saver = tf.train.Saver()
    sess = tf.Session()

    # 对模型变量进行初始化并用其创建会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    checkpoint_dir = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if checkpoint_dir and checkpoint_dir.model_checkpoint_path:
        saver.restore(sess, checkpoint_dir.model_checkpoint_path)
    else:
        print('Not found checkpoint file')
        return False

    batch_xs = cv2.imread(sys.argv[1])
    batch_xs = np.array(batch_xs, dtype=np.float32)
    batch_xs = batch_xs.reshape(1, len(batch_xs), -1, 3)

    start = time.clock()
    predict = sess.run([predict_op], feed_dict={images: batch_xs})
    print(batch_xs.shape)
    output = sess.run(predict_op, feed_dict={images: batch_xs})
    end = time.clock()

    out_path = os.path.join(FLAGS.output_dir, "out.npy")
    np.save(out_path, output)

    print("time: %s\t predict:%.7f" % \
          ((end - start), sum(sum(sum(predict[0])))))

def main(argv=None):
    if gfile.Exists(FLAGS.eval_dir):
        gfile.DeleteRecursively(FLAGS.eval_dir)
    gfile.MakeDirs(FLAGS.eval_dir)

    evaluate()


if __name__ == '__main__':
    tf.app.run()
