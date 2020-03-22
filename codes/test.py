import sys
sys.path.append('../')
import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
from tqdm import tqdm
import options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
args=parser.parse_args()
opt = option.parse(args.opt, is_train=False)
opt = option.dict_to_nonedict(opt)



util.mkdirs(opt['path']['results_root'])
#set log
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)

logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    t=tqdm(range(len(test_loader)))
    for data in test_loader:
        model.feed_data(data)
        image_name=data['LQ_path'][0].split('/')[-1]
        outs=model.test()

        for k, v in outs.items():

            sr_img = util.tensor2img(v)  # uint8
            # save images
            util.mkdir(osp.join(dataset_dir,k))
            save_img_path = osp.join(dataset_dir,k,image_name)
            util.save_img(sr_img, save_img_path)

        t.update()