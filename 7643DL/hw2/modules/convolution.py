import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        m, n_C_prev, n_H_prev, n_W_prev = x.shape
        n_C,n_C_prev,f, f = self.weight.shape
        n_H = int((n_H_prev + 2 * self.padding - f) / self.stride) + 1
        n_W = int((n_W_prev + 2 * self.padding - f) / self.stride) + 1
        Z = np.zeros([m, n_C, n_H, n_W])
        x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=(0, 0))
        for i in range(m):  # loop over the batch of training examples
            a_prev_pad = x_pad[i, :, :, :]  # Select ith training example's padded activation
            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    for c in range(n_C):  # loop over channels (= #filters) of the output volume
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * self.stride
                        vert_end = h * self.stride + f
                        horiz_start = w * self.stride
                        horiz_end = w * self.stride + f

                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end]

                        s = np.multiply(a_slice_prev, self.weight[c,:,:,:])
                        # Sum over all entries of the volume s.
                        z = np.sum(s)
                        Z[i, c, h, w] = z + self.bias[c]

            # Making sure your output shape is correct
        a= (Z.shape == (m, n_C, n_H, n_W))
        out=Z


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        m,n_C_prev, n_H_prev, n_W_prev= x.shape
        n_C,n_C_prev,f, f  = self.weight.shape
        m, n_C, n_H, n_W = dout.shape
        self.dx = np.zeros((m, n_C_prev, n_H_prev, n_W_prev))
        self.dw = np.zeros((n_C, n_C_prev,f, f ))
        self.db = np.zeros(n_C)
        x_pad = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=(0, 0))
        dx_pad = np.pad(self.dx, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=(0, 0))
        for i in range(m):
            a_prev_pad = x_pad[i]
            da_prev_pad = dx_pad[i]
            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    for c in range(n_C):  # loop over the channels of the output volume
                        vert_start = h* self.stride
                        vert_end = vert_start + f
                        horiz_start = w* self.stride
                        horiz_end = horiz_start + f
                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[:,vert_start:vert_end, horiz_start:horiz_end]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[:,vert_start:vert_end, horiz_start:horiz_end] += self.weight[c,:, :, :] * dout[i, c, h, w]
                        self.dw[c,:, :, :] += a_slice * dout[i, c, h, w]
                        self.db[c] += dout[i, c, h, w]

            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            self.dx[i, :, :, :] = da_prev_pad[:,self.padding:-self.padding, self.padding:-self.padding]

            # Making sure your output shape is correct
        a=(self.dx.shape == (m, n_C_prev, n_H_prev, n_W_prev))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################