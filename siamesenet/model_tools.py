import tensorflow as tf
import argparse
import os
import tensorflow.keras.backend as K
from siamesenet.model import SiameseNet

parser = argparse.ArgumentParser(description='')

parser.add_argument("--checkpoint_path", default='/home/tracy/PycharmProjects/SiameseNet/checkpoint/checkpoint_1:4/my_model', help="restore ckpt")  # 原参数路径
parser.add_argument("--org_checkpoint_path", default='/home/tracy/PycharmProjects/SiameseNet/checkpoint/checkpoint_alerted/my_model', help="restore ckpt")  # 原参数路径

parser.add_argument("--new_checkpoint_path", default='/home/tracy/PycharmProjects/SiameseNet/checkpoint/checkpoint_alerted', help="path_for_new ckpt")  # 新参数保存路径
parser.add_argument("--add_prefix", default='siamesenet/', help="prefix for addition")  # 新参数名称中加入的前缀名

args = parser.parse_args()

rename_dict = {
    '_CHECKPOINTABLE_OBJECT_GRAPH': 'checkpoint_object_graph',
    'dense/layer-0/bias/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/dense/fully_connected/biases',
    'dense/layer-0/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/dense/fully_connected/weights',
    'encoder/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/conv1/biases',
    'encoder/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/conv1/weights',
    'encoder/layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization/beta',
    'encoder/layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization/gamma',
    'encoder/layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization/moving_mean',
    'encoder/layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization/moving_variance',
    'encoder/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/conv2/biases',
    'encoder/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/conv2/weights',
    'encoder/layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_1/beta',
    'encoder/layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_1/gamma',
    'encoder/layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_1/moving_mean',
    'encoder/layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_1/moving_variance',
    'encoder/layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/conv3/biases',
    'encoder/layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/conv3/weights',
    'encoder/layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_2/beta',
    'encoder/layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_2/gamma',
    'encoder/layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_2/moving_mean',
    'encoder/layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_2/moving_variance',
    'encoder/layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/conv4/biases',
    'encoder/layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/conv4/weights',
    'encoder/layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_3/beta',
    'encoder/layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_3/gamma',
    'encoder/layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_3/moving_mean',
    'encoder/layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'siamese/encoder/batch_normalization_3/moving_variance'
}


def add_prefix():
    if not os.path.exists(args.new_checkpoint_path):
        os.makedirs(args.new_checkpoint_path)
    with tf.Session() as sess:
        new_var_list = []  # 新建一个空列表存储更新后的Variable变量
        for var_name, _ in tf.contrib.framework.list_variables(args.checkpoint_path):  # 得到checkpoint文件中所有的参数（名字，形状）元组
            var = tf.contrib.framework.load_variable(args.checkpoint_path, var_name)  # 得到上述参数的值

            # if len(var_name.split('/')) > 2:
            #     if var_name.split('/')[2] == 'gamma':
            #         print(var)
            new_name = rename_dict[var_name]  # 在这里加入了名称前缀，大家可以自由地作修改

            # 除了修改参数名称，还可以修改参数值（var）

            print('Renaming %s to %s.' % (var_name, new_name))
            renamed_var = tf.Variable(var, name=new_name)  # 使用加入前缀的新名称重新构造了参数
            new_var_list.append(renamed_var)  # 把赋予新名称的参数加入空列表

        print('starting to write new checkpoint !')
        saver = tf.train.Saver(var_list=new_var_list)  # 构造一个保存器
        sess.run(tf.global_variables_initializer())  # 初始化一下参数（这一步必做）
        model_name = 'model_alterd'  # 构造一个保存的模型名称
        checkpoint_path = os.path.join(args.new_checkpoint_path, model_name)  # 构造一下保存路径
        saver.save(sess, checkpoint_path)  # 直接进行保存
        print("done !")


def display_model():
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(args.org_checkpoint_path):
            if len(var_name.split('/')) > 2:
                var_name_reloaded = rename_dict[var_name]
                var_org = tf.contrib.framework.load_variable(args.org_checkpoint_path, var_name)
                var_reloaded = tf.contrib.framework.load_variable(args.checkpoint_path, var_name_reloaded)
                print(var_name, '||', var_name_reloaded)
                print(var_org - var_reloaded)

            # if len(var_name.split('/')) > 2:
            #     if var_name.split('/')[3] == 'gamma':
            #         var = tf.contrib.framework.load_variable(args.checkpoint_path, var_name)
            #         print(var)


def keras_to_pb():
    K.set_learning_phase(0)
    K.clear_session()
    model = SiameseNet()
    model.load_weights('/home/tracy/PycharmProjects/SiameseNet/checkpoint/with_reference/best/my_model')

    print(model.outputs)
    print(model.inputs)


if __name__ == '__main__':
    display_model()
    # add_prefix()
    # keras_to_pb()