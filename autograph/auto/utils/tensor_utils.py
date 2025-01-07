import numpy as np

# All of these functions assume channels are in the last axis

def im2col(input_tensor, kernal_shape, stride=1, padding=0):
    batch_size, img_height, img_width, num_channels = input_tensor.shape
    filter_height, filter_width = kernal_shape

    # Compute output dimensions
    out_height = (img_height + 2 * padding - filter_height) // stride + 1
    out_width = (img_width + 2 * padding - filter_width) // stride + 1
    # print("[im2col]", out_height, out_width)

    # Add padding to the input
    if padding > 0:
        input_tensor = np.pad(
            input_tensor,
            ((0, 0), (padding, padding), (padding, padding), (0, 0)),
            mode='constant',
            constant_values=0
        )

    # Create the im2col matrix
    col_matrix = np.zeros((batch_size, out_height, out_width, filter_height, filter_width, num_channels))

    # Fill the im2col matrix
    for h in range(filter_height):
        for w in range(filter_width):
            col_matrix[:, :, :, h, w, :] = input_tensor[
                :, 
                h:h + stride * out_height:stride, 
                w:w + stride * out_width:stride,
                :
            ]
    # Reshape to create the column matrix
    col_matrix = col_matrix.transpose(0, 1, 2, 5, 3, 4) 
    col_matrix = col_matrix.reshape(batch_size, out_height * out_width, -1) 
    return col_matrix

def col2im(col_matrix, input_shape, kernal_shape, stride=1, padding=0):
    if len(input_shape) == 4:
        batch_size, img_height, img_width, num_channels = input_shape
    elif len(input_shape) == 3:
        img_height, img_width, num_channels = input_shape
    else: raise Exception("col2im only accepts rank-3 or rank-4 tensors")
    filter_height, filter_width = kernal_shape

    if len(col_matrix.shape) != 3: raise Exception("col matrix needs to have an additional first dimension for the batch_size")
    batch_size, col_out_shape, channels_cross_ker = col_matrix.shape

    # Compute output dimensions
    out_height = (img_height + 2 * padding - filter_height) // stride + 1
    out_width = (img_width + 2 * padding - filter_width) // stride + 1

    # Initialize output tensor with zeros
    padded_height = img_height + 2 * padding
    padded_width = img_width + 2 * padding
    output = np.zeros((batch_size, padded_height, padded_width, num_channels))

    # Reshape col_matrix to extract patches
    # (batch_size, out_height, out_width, filter_height, filter_width, num_channels)
    col_matrix = col_matrix.reshape(batch_size, out_height, out_width, num_channels, filter_height, filter_width)
    col_matrix = col_matrix.transpose(0, 1, 2, 4, 5, 3) 

    # Accumulate gradients for each patch
    for h in range(filter_height):
        for w in range(filter_width):
            output[:, h:h + stride * out_height:stride, w:w + stride * out_width:stride, :] += col_matrix[:, :, :, h, w, :]

    # Remove padding if applied
    if padding > 0:
        output = output[:, padding:-padding, padding:-padding, :]

    return output


def im2col_pool(input_tensor, pool_shape, stride=2):
    N, H, W, C = input_tensor.shape
    pool_h, pool_w = pool_shape
    out_h = (H - pool_h) // stride + 1
    out_w = (W - pool_w) // stride + 1

    col = np.zeros((N, out_h, out_w, pool_h, pool_w, C))

    for y in range(pool_h):
        y_max = y + stride * out_h
        for x in range(pool_w):
            x_max = x + stride * out_w
            col[:, :, :, y, x, :] = input_tensor[:, y:y_max:stride, x:x_max:stride, :]

    col = col.transpose(0, 1, 2, 5, 3, 4).reshape(N, C * out_h * out_w, -1)
    return col

def col2im_pool(col, input_shape, pool_shape, stride=2):
    N, H, W, C = input_shape
    pool_h, pool_w = pool_shape
    out_h = (H - pool_h) // stride + 1
    out_w = (W - pool_w) // stride + 1
    
    col = col.reshape(N, out_h, out_w, C, pool_h, pool_w).transpose(0, 1, 2, 4, 5, 3)
    
    img = np.zeros((N, H, W, C))
    
    for y in range(pool_h):
        y_max = y + stride * out_h
        for x in range(pool_w):
            x_max = x + stride * out_w
            img[:, y:y_max:stride, x:x_max:stride, :] += col[:, :, :, y, x, :]
    
    return img

def reverse_pad(array, pad_width):
    """
    Reverse the effect of np.pad on a numpy array.
    
    Parameters:
    array (ndarray): The padded numpy array.
    pad_width (sequence of tuple): Number of values padded to the edges of each axis.
    
    Returns:
    ndarray: The original array before padding.
    """
    slices = []
    for (before, after) in pad_width:
        start = before
        end = None if after == 0 else -after
        slices.append(slice(start, end))
    
    return array[tuple(slices)]

def batch_numpy_array(dataset: np.ndarray, batch_size: int):
    padding_length = (batch_size - len(dataset) % batch_size) % batch_size
    indices = np.random.choice(dataset.shape[0], size=padding_length)
    padding = dataset[indices]
    padded_dataset = np.concatenate((dataset, padding), axis=0)
    return padded_dataset.reshape((-1, batch_size, *padded_dataset.shape[1:]))


