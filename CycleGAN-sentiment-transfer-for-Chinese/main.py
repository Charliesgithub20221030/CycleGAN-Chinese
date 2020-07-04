import argparse
from cycle_gan import cycle_gan
import tensorflow as tf
# turn off FutureWarning
import warnings
from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def parse():
    # best: 0.5,0.01
    parser = argparse.ArgumentParser(description="cycle GAN")
    parser.add_argument('-model_dir', '--model_dir',
                        default='train_model', help='output model weight dir')
    parser.add_argument('-model_path', '--model_path',
                        help='latest model path')
    parser.add_argument('-batch_size', '--batch_size',
                        default=36, type=int, help='batch size')
    parser.add_argument('-latent_dim', '--latent_dim',
                        default=100, type=int, help='laten size')
    parser.add_argument('-data_source', '--data_source',
                        default='data/feature_twitter_1000.txt', help='data source txt')
    parser.add_argument('-saving_step', '--saving_step',
                        default=1000, type=int, help='saving step')
    parser.add_argument('-num_steps', '--num_steps',
                        default=1000000, type=int, help='number of steps')
    parser.add_argument('-sequence_length', '--sequence_length',
                        default=25, type=int, help='sentence length')
    parser.add_argument('-dis_iter', '--dis_iter', default=3,
                        type=int, help='discriminator iterations')
    parser.add_argument('-load', '--load', action='store_true',
                        help='load pretrained model')
    parser.add_argument('-train', action='store_true', help='whether train')
    parser.add_argument('-test', action='store_true', help='whether test')
    parser.add_argument('-file_test', action='store_true',
                        help='whether test input file')
    parser.add_argument('-mode', default='all', help='type pretrain or all')
    parser.add_argument('-dict', '--dict', default='word2vec_model/dictionary.json',
                        help='dictionary for tokenization')
    parser.add_argument('-feature_file', '--feature_file',
                        default=False, help='prediction batch from feature file')
    args = parser.parse_args()
    return args


def run(args):
    sess = tf.Session()
    if args.test:
        args.mode = 'test'
    model = cycle_gan(args, sess)
    if args.train and args.mode == 'all':
        model.train()
    if args.train and args.mode == 'pretrain':
        model.pretrain()
    if args.file_test:
        model.file_test()
    if args.test:
        model.test()


if __name__ == '__main__':
    args = parse()
    run(args)
