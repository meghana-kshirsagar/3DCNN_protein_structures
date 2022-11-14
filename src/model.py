import torch
import torch.nn as nn


# Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# in_channels = 4, 

class Pockets3DCNN(nn.Module):
    def __init__(self, input_shape, filters_shape, filter_sizes, n_outs):
        super(Pockets3DCNN, self).__init__()

        (in_channels, in_time, in_height, in_width) = input_shape
        (filt_time, filt_height, filt_width) = filters_shape

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=filter_sizes[0], 
                               kernel_size=(filt_time,filt_height,filt_width), padding=1)
        self.conv2 = nn.Conv3d(in_channels=filter_sizes[0], out_channels=filter_sizes[1], 
                               kernel_size=(filt_time,filt_height,filt_width), padding=1)
        self.conv3 = nn.Conv3d(in_channels=filter_sizes[1], out_channels=filter_sizes[2], 
                               kernel_size=(filt_time,filt_height,filt_width)) #, padding=0)

        self.bn1 = nn.BatchNorm3d(filter_sizes[0]) 
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2))   # stride=Default value is kernel_size
        
        n_out_penultimate = filter_sizes[-1] * (in_time/(2**2) - filt_time+1) * (in_height/(2**2) - filt_height+1) *(in_width/(2**2) - filt_width+1)  ### no padding in last Conv3d
        print('N-out penultimate:',n_out_penultimate)
        
        self.fc = nn.Linear(in_features=int(n_out_penultimate), out_features=n_outs, bias=True)
        
        
    def reset_parameters(self):
        print('Resetting parameters ...')
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.bn1.reset_parameters()
        self.fc.reset_parameters()
        
        print('Initializing parameters ...')
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.xavier_normal_(self.conv3.weight)


    # LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
    def forward(self, x):
        dropout_prob = 0.7

        x = self.conv1(x)
        #x = self.bn1(x)
        #x = nn.LeakyReLU(negative_slope=0.1)(x)
        x = self.pool(x)
        x = nn.Dropout(p=dropout_prob)(x)

        x = self.conv2(x)
        #x = nn.LeakyReLU(negative_slope=0.1)(x)
        x = self.pool(x)
        x = nn.Dropout(p=dropout_prob)(x)

        x = self.conv3(x)
        #x = nn.LeakyReLU(negative_slope=0.1)(x)
        x = nn.Dropout(p=dropout_prob)(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        #x = nn.LogSoftmax(dim=1)(x)
        #print(x.shape)
        
        return x
    
def get_old_params_3D_CNN():
    in_channels = 4
    in_time = 20
    in_height = 20
    in_width = 20
    
    filt_time = 3
    filt_height = 3
    filt_width = 3
    
    filter_sizes=[32, 64, 128]
    #filter_sizes=[16, 32, 64]

    input_shape = (in_channels, in_time, in_height, in_width)
    filters_shape = (filt_time, filt_height, filt_width)
    
    return input_shape, filters_shape, filter_sizes


def get_params_3D_CNN():
    params = dict()
    # define training hyperparameters
    params['learning_rate']=1e-4
    params['feat_dim'] = 128
    params['out_dim'] = 1
    # 3DCNN feature extraction module
    params['in_dim'] = 4  ## num channels
    params['box_size'] = 20

    return params


class CNN3D(nn.Module):
    """Base 3D convolutional neural network architecture for molecular data, consisting of six layers of 3D convolutions, each followed by batch normalization, ReLU activation, and optionally dropout.
    This network uses strided convolutions to downsample the input twice by half, so the original box size must be divisible by 4 (e.g. an atomic environment of side length 20 Å with 1 Å voxels). The final convolution reduces the 3D box to a single 1D vector of length ``out_dim``.
    The desired input and output dimensionality must be specified when instantiating the model.

    :param in_dim: Input dimension.
    :type in_dim: int
    :param out_dim: Output dimension.
    :type out_dim: int
    :param box_size: Size (edge length) of voxelized 3D cube.
    :type box_size: int
    :param hidden_dim: Base number of hidden units, defaults to 64
    :type hidden_dim: int, optional
    :param dropout: Dropout probability, defaults to 0.1
    :type dropout: float, optional
    """
#Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    def __init__(self, in_dim, out_dim, box_size, hidden_dim=64, dropout=0.1):
        super(CNN3D, self).__init__()
        self.out_dim = out_dim

        kernel_size = int(box_size/4)
        self.model = nn.Sequential(
            nn.Conv3d(in_dim, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv3d(hidden_dim, hidden_dim * 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv3d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv3d(hidden_dim * 4, hidden_dim * 8, 3, 1, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv3d(hidden_dim * 8, hidden_dim * 16, 3, 1, 1, bias=False),
            nn.BatchNorm3d(hidden_dim * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv3d(hidden_dim * 16, out_dim, kernel_size, 1, 0, bias=False),
        )


    def forward(self, input):
        """Forward method.

        :param input: Input data, as voxelized 3D cube of shape (batch_size, in_dim, box_size, box_size, box_size).
        :type input: torch.FloatTensor
        :return: Output of network, of shape (batch_size, out_dim)
        :rtype: torch.FloatTensor
        """
        bs = input.size()[0]

        output = self.model(input)
        return output.view(bs, self.out_dim)
