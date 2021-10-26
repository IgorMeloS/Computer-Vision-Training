# =============================================================================
# learning rate scheduler functions
# =============================================================================
import numpy as np
class LRFunc:
    """Class with learn rate scheculer functions
    Args:
        lr: initial value of learning rate
        epochs: total number of epochs
        step: number of step to dropdown the learning rate (just for step_decay function)
        factor: factor decay for the step_decay, float number between 0 and 1.
        degree: degree value of the polynomial function to dropdown the learning rate (valid for the poly_decay function)
    """
    def __init__(self, l_r = 0.001, epochs = 1000, step = 10, factor = 0.99, degree = 1):
        self.l_r = l_r
        self.epochs = epochs
        self.step = step
        self.factor = factor
        self.degree = degree
    def step_decay(epoch, lr):

        init_alpha = 0.001 # initial learning rate

         # step size
        if epoch  == 0:
            lr = init_alpha
            print("The intitial learning rate is {:f}".format(lr))
        elif   epoch % self.step  == 0:
            tol = 1e-6
            lr = init_alpha * (self.factor ** np.floor((1 + epoch) / self.step)) + tol
            print("Learning rate updated {: f}".format(lr))
        else:
            lr = lr
        return lr

    def poly_decay(epoch):

        alpha = self.l_r * (1 - (epoch / float(self.epochs))) ** self.degree
        if   epoch % self.step  == 0:
            print("Learning rate {: f}".format(alpha))
        # return the new learning rate
        return alpha
