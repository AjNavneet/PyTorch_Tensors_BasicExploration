import torch

class Arithmetic:
    def operations(self, x):
        # ### Add and Subtract

        # Add two tensors
        x2 = torch.add(x, x)
        print(x2)
        print('x2 shape: {}\n'.format(x2.shape))

        # Alternatively, you can use the '+' operator for addition
        x2 = x + x
        print(x2)
        print('x2 shape: {}'.format(x2.shape))

        # Subtract one tensor from another
        x_2 = torch.subtract(x, x)
        print(x_2)
        print('x_2 shape: {}\n'.format(x_2.shape))

        # Alternatively, you can use the '-' operator for subtraction
        x_2 = x - 2 * x
        print(x_2)
        print('x_2 shape: {}'.format(x_2.shape))

        # ### Multiply

        # Element-wise multiplication
        x_mult = x[0:5, :] * x[0:5, :]
        print(x_mult.shape)
        print(x_mult)

        # Matrix multiplication of a sliced tensor with its transpose
        x_matmult = x[0:5, :] @ x[0:5, :].T
        print(x_matmult.shape)
        # 5 X 3 | 3 X 5
        print(x_matmult)

        # ### Trigonometric Operations

        # All trigonometric functions can be performed on tensors (values in radians)
        x_cos = torch.cos(x[:5, :])
        print(x_cos)

        x_sin = torch.sin(x[:5, :])
        print(x_sin)

        # ### Statistics

        # Mathematical operations for statistics

        # Calculate the mean value of the number of weekly_mins_watched and minimum daily mins
        x_mean = x.mean(dim=0)
        print(x_mean)

        # Calculate the median value of the number of weekly_mins_watched and minimum daily mins
        x_median = x.median(dim=0)
        print(x_median)
