import models.arch.RRDBNet_arch as RRDBNet_arch




def define_SR(opt):


    opt_net = opt['network_SR']


    which_model = opt_net['which_model_G']

    #SR

    if which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    else:
        raise NotImplementedError('Generator models [{:s}] not recognized'.format(which_model))

    return netG


