from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                v0=self.grad_tracker[idx]["dw"]
                v_n=self.momentum*v0-self.learning_rate*m.dw
                m.weight=m.weight+v_n
                self.grad_tracker[idx]["dw"]=v_n
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                # pass
                v0 = self.grad_tracker[idx]["db"]
                v_n = self.momentum * v0 - self.learning_rate * m.db
                m.bias = m.bias + v_n
                self.grad_tracker[idx]["db"] = v_n
                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
