import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''

        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        (m, n_C_prev, n_H_prev, n_W_prev) = x.shape
        n_H = int(1 + (n_H_prev - self.kernel_size) / self.stride)
        n_W = int(1 + (n_W_prev - self.kernel_size) / self.stride)
        n_C = n_C_prev
        out = np.zeros((m, n_C, n_H, n_W))
        for i in range(m):  # loop over the training examples
            for h in range(n_H):  # loop on the vertical axis of the output volume
                for w in range(n_W):  # loop on the horizontal axis of the output volume
                    for c in range(n_C):  # loop over the channels of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * self.stride
                        vert_end = h * self.stride + self.kernel_size
                        horiz_start = w * self.stride
                        horiz_end = w * self.stride + self.kernel_size

                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = x[i, c, vert_start:vert_end, horiz_start:horiz_end]

                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                        out[i, c, h, w] = np.max(a_prev_slice)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        H_out=n_H
        W_out=n_W
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        m, n_C_prev, n_H_prev, n_W_prev = x.shape
        m, n_C, n_H, n_W = dout.shape
        self.dx = np.zeros(x.shape)
        # self.dx = dout
        for i in range(m):  # loop over the training examples
            # select training example from A_prev (≈1 line)
            a_prev = x[i]
            for c in range(n_C):
                for h in range(n_H):  # loop on the vertical axis
                    for w in range(n_W):  # loop on the horizontal axis
                      # loop over the channels (depth)
                        # Find the corners of the current "slice" (≈4 lines)
                        # vert_start = h
                        vert_start = h*self.stride
                        vert_end = vert_start + self.kernel_size
                        # horiz_start = w
                        horiz_start = w* self.stride
                        horiz_end = horiz_start + self.kernel_size

                        # Compute the backward propagation in both modes.
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[c,vert_start:vert_end, horiz_start:horiz_end]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = a_prev_slice == np.max(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        a=dout[i, c, h, w]
                        # a=dout[i, c, vert_start:vert_end, horiz_start:horiz_end]
                        self.dx[i, c, vert_start:vert_end, horiz_start:horiz_end] += np.multiply(mask,a)
        a=(self.dx.shape == x.shape)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
