import os
import os.path as osp
import logging
import yaml
from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()


def parse(opt_path, is_train=True):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)


    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    opt['is_train'] = is_train
    scale = opt['scale']

    # datasets
    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase
        dataset['scale'] = scale

        is_lmdb=False
        if dataset.get('dataroot_GT', None) is not None:
            if dataset['dataroot_GT'].endswith('lmdb'):
                is_lmdb = True
        if dataset.get('dataroot_LQ', None) is not None:
            dataset['dataroot_LQ'] = osp.expanduser(dataset['dataroot_LQ'])
            if dataset['dataroot_LQ'].endswith('lmdb'):
                is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'



    # path
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_state'] = osp.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['tb'] = osp.join(experiments_root, 'tb')
        opt['path']['val_images'] = osp.join(experiments_root, 'val_images')




    else:  # test
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        results_root = osp.join(experiments_root, 'results')
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root


    print('option loading done')
    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')


    if opt['path']['resume_state']:

        if opt['path'].get('pretrain_model_G_AB', None) is not None or opt['path'].get(
                'pretrain_model_D_A', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path']['pretrain_model_G_AB'] = osp.join(opt['path']['models'],
                                                      '{}_G_AB.pth'.format(resume_iter))
        logger.warning('Set [pretrain_model_G_AB] to ' + opt['path']['pretrain_model_G_AB'])

        if 'cycle' in opt['model']:

            opt['path']['pretrain_model_G_BA'] = osp.join(opt['path']['models'],
                                                          '{}_G_BA.pth'.format(resume_iter))

            logger.warning('Set [pretrain_model_G_BA] to ' + opt['path']['pretrain_model_G_BA'])

        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D_A'] = osp.join(opt['path']['models'],
                                                       '{}_D_A.pth'.format(resume_iter))
            logger.warning('Set [pretrain_model_D_A] to ' + opt['path']['pretrain_model_D_A'])
            if 'cycle' in opt['model']:
                opt['path']['pretrain_model_D_B'] = osp.join(opt['path']['models'],
                                                             '{}_D_B.pth'.format(resume_iter))


                logger.warning('Set [pretrain_model_D_B] to ' + opt['path']['pretrain_model_D_B'])


