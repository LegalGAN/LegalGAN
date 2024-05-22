import hydra
import os
import sys
# Change path to LegalGAN directory
sys.path.append('/home/brock_carey1/LegalGAN')

from data_utils.dataset import load_data
from train_utils.trainer import TrainLoop

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

@hydra.main(version_base=None, config_path="../", config_name="config")
def main(config):
    import tensorflow as tf; print(tf.config.experimental.list_physical_devices('GPU'))
    data = load_data(
        config=config
    )
    with tf.distribute.MirroredStrategy().scope():
        TrainLoop(
            config=config,
            data=data
        ).run_loop()
    
if __name__ == '__main__':
	main()
