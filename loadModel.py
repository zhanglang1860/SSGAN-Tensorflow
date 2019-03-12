import tensorflow as tf
from VAE import *

class LoadModel(object):
    def __init__(self,batchsize, n_z):
        self.batchsize = batchsize
        self.n_z = n_z

        self.saver = tf.train.import_meta_graph(
            '/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-encoder-Tensorflow/preTrainedModel/train-9.meta'
        )
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = tf.Session(config=session_config)

    def load_pre_train_model(self, input):
        self.saver.restore(self.session, tf.train.latest_checkpoint(
            '/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-encoder-Tensorflow/preTrainedModel'))

        image_matrix = tf.reshape(input, [-1, 28, 28, 1])
        vae_model = LatentAttention(self.batchsize,self.n_z)
        z_mean, z_stddev = vae_model.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize, self.n_z], 0, 1, dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)
        print ("Load model complete.")
        return guessed_z







# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph(
#         '/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-encoder-Tensorflow/preTrainedModel/train-9.meta'
#     )
#     saver.restore(sess,tf.train.latest_checkpoint('/media/wenyu/8d268d3e-37df-4af4-ab98-f5660b2e71a7/wenyu/PycharmProjects/SSGAN-encoder-Tensorflow/preTrainedModel'))

    # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    #
    # tensor_name_list = [t.name for op in tf.get_default_graph().get_operations() for t in op.values()]
    #
    # for tensor_name in tensor_name_list:
    #     print(tensor_name, '\n')


# tvs = [v for v in tf.trainable_variables()]
#         for v in tvs:
#             print(v.name)
#             print(self.session.run(v))

    # gv = [v for v in tf.global_variables()]
    # for v in gv:
    #     print(v.name)
    #
    # ops = [o for o in sess.graph.get_operations()]
    # for o in ops:
    #     print(o.name)


    # print sess.run(tf.get_default_graph().get_tensor_by_name("guessed_z:0"))



    # with tf.get_default_graph() as graph:
    #     w_mean = dense(h2_flat, 7 * 7 * 32, self.n_z, "w_mean")
    #     w_stddev = dense(h2_flat, 7 * 7 * 32, self.n_z, "w_stddev")
    #     w_mean = dense(h2_flat, 7 * 7 * 32, self.n_z, "w_mean")
    #     w_stddev = dense(h2_flat, 7 * 7 * 32, self.n_z, "w_stddev")
    #   data = graph.get_tensor_by_name('data:0')
    #   output = graph.get_tensor_by_name('output:0')
    #   logits = tf.nn.softmax(output)