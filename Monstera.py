

import math
import tensorflow as tf





# TODO: use a more industry standard way to warn the user of invalid behavior (instead of just print....)
# TODO: Add better input validation
# TODO: Add Unit Tests
# TODO: make it work with multiple versions of TF


# Utilities

def calculate_groupwise_splits(filters, min_splits = 3, max_splits = 128):
    """
    This calculates the number of groupwise splits that should be used for the provided filters count.
    In some cases an alternative filters count value will be returned, if the provided filters count was not usable.

    Note: we can only use GPU accelerated Groupwise Convolutions if the number of input and output filters is a multiple of the number of groups it's split into.

    Args:
        filters: (int) This is the count of filters that is being grouped.
        min_splits: (int)(default: 3) This is the (exclusive) minimum number of splits to consider. It will force the chosen splits value to be larger than this value.
        max_splits: (int)(default: 128) This is the (inclusive) maximum number of splits to consider. It will force the chosen splits value to be less than or equal to this value.

    Returns:
        This returns two integers, the first is the filters count that was selected, and second is the number of splits that was selected.
        Note: the returned filters count can be greater than the provided value if no sutible splits was found for the provided value.

    """

    # Basic check of the inputs for validity
    filters = int(filters)
    min_splits = int(min_splits)
    max_splits = int(max_splits)
    if(filters <= 0 or min_splits <= 0 or max_splits <= 0):
        print("Input Values cannot be 'Less Than Or Equal To Zero'")
        return (None, None)

    # For our uses, we do not want the larger half of the Factors, so, by starting at the square root will give us the lower half of possible factors.
    splits = math.floor(math.sqrt(filters))

    # If a large filters count was passed in, resulting in a splits that exceeds Max, then use max_splits instead
    if (max_splits < splits):
        # warn the user that the data triggered a special condition.
        print("Calculated Splits ["+str(splits)+"] Exceeds Max Splits ["+str(max_splits)+"], using Max Splits value.")
        splits = max_splits

    # Check if filters is divisible by splits, if so, we're done.
    if(0 == (filters % splits)):
        return (filters, splits)
    # if it's not divisible, then check for other smaller factors
    else:
        
        # Very small factors, like 1, 2, or 3 are not useful in this case, so, set a lower bounds at half the split value
        half_splits = math.floor(splits / 2) # Used 'floor()' because 'range()' will not include the value itself.

        # If the lower bounds is below the provided minimum value, use the minimum instead. 
        if (half_splits < min_splits):
            # warn the user that the data triggered a special condition.
            print("Calculated Half Splits ["+str(half)+"] is Below Min Splits ["+str(min_splits)+"], using Min Splits value.")
            half_splits = min_splits

        # Brute force try them one at a time going from the largest to smallest.
        for try_factor in range(splits, half_splits, -1):
            # Check if the value is a Factor, if so, then we are done, return that.
            if (0 == (filters % try_factor)):
                return (filters, try_factor)

        # We may not always find a reasonable factor (filters count could be a prime number) so, we increment it and then call this recursively. 
        return calculate_groupwise_splits((filters + 1), min_splits, max_splits)




# Base Class Implementation

class MonsteraBase(tf.keras.layers.Layer):
    """

    This Convolution Layer takes the input filters and first splits them into groupwise convolutions.
    After the groupwise convolution this interleaves the results before applying a second Groupwise Convolution.
    This is done to reduce the number of MACCs (MADDs, Flops) that the layer must compute, as well as resulting in fewer parameters being used.
    
    
    This is named after a plant with split leaves (https://en.wikipedia.org/wiki/Monstera)
    (Because it splits the filters and interleaves the outputs, get it? split leaves.... ugh, fine.... you're no fun.)


    The basic process is very similar to how shuffelnet works (https://arxiv.org/abs/1707.01083).
    Key differences are:
        This allows every output to be influnced by every* input.
        This acts as a single layer, and is seperate from the depthwise operation.
        This can take any number of input filters and can output any number of filters. (even prime numbers)
        This can utilize GPU acceleration on some GPUs.


    Implementation Concerns:
        Channels First is recomended for performance reasons (but results may vary)
        For small numbers of filters, this may not reduce the number of computations required.
        This layer uses fewer parameters and when combined with Depthwise Convolutions, may cause the network to be under-paramertized.
        For any Network that is memory bottlenecked, like MobilenetV2, this will likely perform worse when using the Default TF implementation.
        The GPU implementation is only supported by certain CUDA (CUDNN) Versions, read more here (https://github.com/tensorflow/tensorflow/pull/25818)
        Depending on the number of input Filters, and the number of output filters, some performance may be wasted (Zero-Padded inputs, and Truncated Outputs)


    NOTE: These arguments are set to match the Conv layer in Keras (https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/convolutional.py#L48)
        Most of these areguments or directly passed throght to the underlying Conv Implementation.
    NOTE: The GPU implementation may not support all arguments. Note, the GPU implementation cannot run on CPU.

    Arguments:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space (i.e. the number
          of filters in the convolution).
        

        depth_multiplier: The multiple of the internal representation between the 2 groupwise layers. 
            This functions similarly to depth_multiplier in SeparableConv2D in that it increase the internal representation.
            Note: this increase the accuracy at the cost of increased computations, # of Params, and Memory Loading.
        min_splits: The minimum number of splits to be used by this layer 
        max_splits: the maximum number of splits to be used by this layer 
        
        
        kernel_size: An integer or tuple/list of n integers, specifying the
          length of the convolution window.
        strides: An integer or tuple/list of n integers,
          specifying the stride length of the convolution.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
        padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
        data_format: A string, one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, ..., channels)` while `channels_first` corresponds to
          inputs with shape `(batch, channels, ...)`.
        dilation_rate: An integer or tuple/list of n integers, specifying
          the dilation rate to use for dilated convolution.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any `strides` value != 1.
        activation: Activation function. Set it to None to maintain a
          linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: An initializer for the convolution kernel.
        bias_initializer: An initializer for the bias vector. If None, the default
          initializer will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The function
            must take as input the unprojected variable and must return the
            projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` the weights of this layer will be marked as
          trainable (and listed in `layer.trainable_weights`).
        name: A string, the name of the layer.


    """

    def __init__(self, 
                 rank,
                 filters,

                 #use_gpu_impl = False
                 depth_multiplier = 1,
                 min_splits = 3, 
                 max_splits = 128,
                 

                 kernel_size=1,
                 strides=1,
                 padding="same", 
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 dtype=None,
                 **kwargs):

        super(MonsteraBase, self).__init__(
                trainable=trainable,
                name=name,
                activity_regularizer=regularizers.get(activity_regularizer),
                dtype=dtype,
                dynamic=False,
                **kwargs)


        # Store all the values for constructing the Convolutional Layers
        self.rank = rank
        self.filters = filters
        self.dtype = dtype

        # this is basically taken verbatim from the normal Convolution at (https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/convolutional.py#L99)
        self.kernel_size = conv_utils.normalize_tuple(
                kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if (self.padding == 'causal' and not isinstance(self,
                                                        (Conv1D, SeparableConv1D))):
          raise ValueError('Causal padding is only supported for `Conv1D`'
                           'and ``SeparableConv1D`.')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


        # Save layer specific values
        self.depth_multiplier = int(depth_multiplier)
        self.min_splits = int(min_splits)
        self.max_splits = int(max_splits)


        # Default layer specific values.
        self.zero_pad_inputs = None # Holds the value of how much the inputs should be zero padded
        self.input_splits = None # holds the number of splits for the Input Layer
        self.input_split_depth = None # holds the number of items in each split
        self.truncate_output = None # If the Outpur should be truncated and if so, by how much
        self.output_splits = None # holds the number of splits for the Output Layer
        self.output_split_depth = None # holds the number of items in each split


        self.input_rank = None # The number of dimensions of the input, including batch (2D should be 4: batch, x, y, channels)
        self.input_channels = None # The number of Channels in the Input data
        self.input_shape = None # Stores a copy of the input dimensions for future use.

        return

    def build(self, input_shape):
        """
        Creates the variables of the layer (optional, for subclass implementers).
        This is a method that implementers of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.
        This is typically used to create the weights of `Layer` subclasses.
        Arguments:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).
        """

        # Bits of this are taken from TF.Keras source : https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/convolutional.py#L146
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = -1
        if input_shape.dims[self.channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                    'should be defined. Found `None`.')
        input_dim = int(input_shape[self.channel_axis])
        
        # Define the Conv Padding
        if self.padding == 'causal':
            self.op_padding = 'valid'
        else:
            self.op_padding = self.padding
        if not isinstance(self.op_padding, (list, tuple)):
            self.op_padding = self.op_padding.upper()

        # build the input_shape as a shape Array of ints
        shape = []
        # loop through each dimension
        for dim in list(input_shape):
            if(dim is None or dim == None or (False == isinstance(dim, int) and dim.value == None)):
                shape.append(None)
            elif(isinstance(dim, int)):
                shape.append(int(dim))
            else:
                shape.append(int(dim.value))

        # Store values for later use
        self.input_rank = len(shape)
        self.input_shape = shape.copy()

        # Gets the number of input channels and sets the Filters on the shape for output.
        if (self.data_format == 'channels_last'):
            self.input_channels = int(shape[-1])
            shape[-1] = self.filters
        elif (self.data_format == 'channels_first'):
            self.input_channels = int(shape[1])
            shape[1] = self.filters
        else:
            # can't do anything if it's not Channels first or last since it has no channels
            raise Exception('The data_format provided did not have an appropriate value of either "channels_last" or "channels_first" ')
        
        # Calculate the internal sizes
        # Use the utility method to calculate the input splits and possible padding
        (input_size, input_splits) = calculate_groupwise_splits(self.input_channels, self.min_splits, self.max_splits)
        self.zero_pad_inputs = input_size - self.input_channels # if there is a difference, this is how many zero padded values should be added to the input channels
        self.input_splits = input_splits
        self.input_split_depth = int(input_size / input_splits) # this should result in a whole number

        # Use the utility method to calculate the output splits and possible truncation
        (output_size, output_splits) = calculate_groupwise_splits(self.filters, self.min_splits, self.max_splits)
        self.truncate_output = output_size - self.filters # if there is a difference, this is how many values that should be truncated from the output channels
        self.output_splits = output_splits
        self.output_split_depth = int(output_size / output_splits) # this should result in a whole number

        # calculate the Channels count of the intermidiate layer with the interleaved results.
        self.inter_channels_count = self.input_splits * self.output_splits * self.depth_multiplier

        # Define the Indices for the interleaving of outputs
        self.interleave_indices = []
        
        # Loop through the different inputs and build the indexes
        for index in range(0, int(self.output_splits)): 
            for inter in range(0, int(self.input_splits)):
                for multiply in range(0, self.depth_multiplier):
                    # This is to make sure that each of the output groups recieves values from each input group
                    start_idx = index 
                    idx_offset = inter * self.output_splits * self.depth_multiplier 
                    mult_offset = multiply * self.output_splits
                    self.interleave_indices.append(  start_idx + idx_offset + mult_offset) 



        print("self.interleave_indices")
        print(self.interleave_indices)
        print("That's all folks")



        # make the Values a TF constant for use in the graph
        self.interleave_indices = tf.constant(self.interleave_indices)

        # Determine the size of the weights (kernel) matrix
        # The Input Layers weights should be `input_splits` matrixes of size `input_split_depth` by `output_splits` by `self.depth_multiplier`
        self.input_kernel_shape = self.kernel_size + (int(self.input_split_depth), (int(self.output_splits) * int(self.input_splits) * self.depth_multiplier))
        self.input_bias_shape = (int(self.output_splits) * int(self.input_splits) * self.depth_multiplier)

        # The Output Layers weights should be `output_splits` matrixes of size `input_splits`by `self.depth_multiplier` by `output_split_depth`
        self.output_kernel_shape = self.kernel_size + ((int(self.input_splits) * self.depth_multiplier), (int(self.output_splits) * int(self.output_split_depth))),
        self.output_bias_shape = (int(self.output_splits) * int(self.output_split_depth))

        # Build the weight matrixes
        # Input Layer Weights
        self.input_layer_grouped_weights = self.add_weight(name='input_layer_grouped_weights', 
                                      shape = input_kernel_shape,
                                      initializer = self.kernel_initializer,
                                      regularizer = self.kernel_regularizer,
                                      constraint = self.kernel_constraint,
                                      trainable = True,
                                      dtype = self.dtype)

        # TODO: there should be some experiments to see if this performs better with or without bias in the middle
        if self.use_bias:
            self.input_bias = self.add_weight(name='input_layer_grouped_bias',
                            shape=(self.input_bias_shape, ),
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                            trainable=True,
                            dtype=self.dtype)
        else:
            self.input_bias = None

        # Output Layer Weights
        self.output_layer_grouped_weights = self.add_weight(name='output_layer_grouped_weights', 
                                      shape = output_kernel_shape,
                                      initializer = self.kernel_initializer,
                                      regularizer = self.kernel_regularizer,
                                      constraint = self.kernel_constraint,
                                      trainable = True,
                                      dtype = self.dtype)

        if self.use_bias:
            self.output_bias = self.add_weight(name='output_layer_grouped_bias',
                            shape=(self.output_bias_shape, ),
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                            trainable=True,
                            dtype=self.dtype)
        else:
            self.output_bias = None

        # set up the shape and size of the zero paddings, if needed
        if(self.zero_pad_inputs > 0):
            
            if (self.data_format == 'channels_last'):
                pad = []
                # Add blank padding for all dimensions except last
                for i in range(1, self.input_rank):
                    pad.append([0,0])
                # The last dimension is the Channels, so we add the padding for that
                pad.append([0, self.zero_pad_inputs])
                # This should pad the input Channels dimension to make it work for the Grouped convolution
                self.zero_paddings = tf.constant(pad)

            elif (self.data_format == 'channels_first'):
                # This should pad the input Channels dimension to make it work for the Grouped convolution
                self.zero_paddings = tf.constant([[0, 0,], [0, self.zero_pad_inputs]])




        # Build the Input Layer Convolution
        # TODO: this is unlikely to work due to the constraints on the Groupwise Convolution Filter shape, but maybe, so, I'm going to try it anyway 
        self.input_convolution_op = nn_ops.Convolution(
                    input_shape,
                    filter_shape = self.input_layer_grouped_weights.shape,
                    dilation_rate = self.dilation_rate,
                    strides = self.strides,
                    padding = self.op_padding,
                    data_format = conv_utils.convert_data_format(self.data_format, self.rank + 2))

        # calculate the shape of the Interleaved Intermediate results
        inter_shape = input_shape.copy()
        inter_shape[self.channel_axis] = self.inter_channels_count 

        # Build the Output Layer Convolution
        # TODO: this is unlikely to work due to the constraints on the Groupwise Convolution Filter shape, but maybe, so, I'm going to try it anyway 
        self.output_convolution_op = nn_ops.Convolution(
                    inter_shape, 
                    filter_shape = self.output_layer_grouped_weights.shape,
                    dilation_rate = 1,
                    strides = 1,
                    padding = "SAME",
                    data_format = conv_utils.convert_data_format(self.data_format, self.rank + 2))



        # Done, make as built and return the output shape
        self.built = True

        return tuple(shape)


    def call(self, inputs):

        # Bits of this are taken from TF.Keras source : https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/convolutional.py#L196
        # It gets duplicated twice because of the Input Layer/Output Layer relationship
        
        # Run the CNN Layer
        outputs = self.input_convolution_op(inputs, self.input_layer_grouped_weights)

        # Apply the Bias 
        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.input_bias, (1, self.input_bias_shape, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.input_bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.input_bias, data_format='NHWC')
        
        # Apply the Activation
        if self.activation is not None:
            outputs = self.activation(outputs)

        # Interleave the results using tf.gather (https://www.tensorflow.org/api_docs/python/tf/gather)
        outputs = tf.gather(
            outputs,
            self.interleave_indices,
            axis=self.channel_axis,
            batch_dims=0
            )
        
        # Run the CNN Layer
        outputs = self.output_convolution_op(outputs, self.output_layer_grouped_weights)

        # Apply the Bias 
        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.output_bias, (1, self.output_bias_shape, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.output_bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.output_bias, data_format='NHWC')
        
        # Apply the Activation
        if self.activation is not None:
            outputs = self.activation(outputs)

        # Truncate extra outputs if nessisary
        if self.truncate_output > 0:
            if (self.data_format == 'channels_last'):
                # I wish I knew a cleaner way to do this.
                if self.rank == 1:
                    outputs = outputs[:, :, 0:(-self.truncate_output)]
                elif self.rank == 2:
                    outputs = outputs[:, :, :, 0:(-self.truncate_output)]
                elif self.rank == 3:
                    outputs = outputs[:, :, :, :, 0:(-self.truncate_output)]
                elif self.rank == 4:
                    outputs = outputs[:, :, :, :, :, 0:(-self.truncate_output)]
            # Truncate extra channels is easy for channels first.
            elif (self.data_format == 'channels_first'):
                outputs = outputs[:, 0:(-self.truncate_output)]

        # Finally return the outputs
        return outputs






# 1D Convolution Implementation 
# TODO: Implement this later


# 2D Convolution Implementation 
# for now, just use Base and set `rank=2,`






# 3D Convolution Implementation 
# TODO: Implement with this later 

