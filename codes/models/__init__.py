import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration

    if model == 'SR':  # unsupervised real-world super-resolution
        from .RRDBSR_model import RRDBSRModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))

    m = M(opt)


    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
