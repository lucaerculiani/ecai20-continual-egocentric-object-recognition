import logging


class DynamicOptimizer(object):

    def __init__(self, base_optimizer):

        super(DynamicOptimizer, self).__init__()

        self.opt = base_optimizer
        self.zero_grad = self.opt.zero_grad
        self.step = self.opt.step
        self.state_dict = self.opt.state_dict
        self.load_state_dict = self.opt.load_state_dict

    def update_state(self, *args):
        pass

    @property
    def logger(self):
        return logging.getLogger(__name__)


class UpdateableOptimizer(DynamicOptimizer):

    def __init__(self, base_optimizer, update_function):

        super(UpdateableOptimizer, self).__init__(base_optimizer)

        self.update_function = update_function

    def update_state(self, iteration):
        self.update_function(iteration, self.opt)


class DecayingLROptimizer(UpdateableOptimizer):
    def __init__(self, base_optimizer, rate=0.1):

        self.rate = rate

        super(DecayingLROptimizer, self).__init__(base_optimizer,
                                                  self.step_decay)

    def step_decay(self, iteration, opt):

        self.logger.debug("iteration {}, step decay of {}".format(iteration,
                                                                  self.rate))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * self.rate
