from torch import nn
import torch 
import torchvision.transforms.functional as F

#------------------------------ BubLeNet --------------------------------------
#Defining the convolutional neural network 
class BubLeNet(nn.Module):                         # Extending nn.Module class 
    def __init__(self, num_classes):                         # Constructor 
        super(BubLeNet,self).__init__()            # Calls the constructor of nn.Module
        self.cnn_net = nn.Sequential(         # nn.Sequentila allows multiple layers to stack together
            # nn.Conv2d(1,6,5),                   #(N,1,28,28) -> (N,6,24,24)
            nn.Conv2d(3,6,18,stride=2),         #(N,3,64,64) -> (N,6,24,24)
            nn.Tanh(),                      
            nn.AvgPool2d(2,stride=2),           #(N,6,24,24) -> (N,6,12,12)
            nn.Conv2d(6,16,5),                  #(N,6,12,12) -> (N,16,8,8)
            nn.Tanh(),
            nn.AvgPool2d(2,stride=2)            #(N,16,8,8) -> (N,16,4,4)
            )
        
        self.fc_net = nn.Sequential(          # Fully connected layer 
            nn.Linear(256,120),
            nn.Tanh(),
            nn.Linear(120,84),
            nn.Tanh(),
            # nn.Linear(84,10)
            nn.Linear(84, num_classes)
            )
        
# It get a batch of data which have defined earlier 
        
    def forward(self,x):     
        #print(x.shape)
        x = self.cnn_net(x)       
        #print(x.shape)
        x = x.view(x.size(0),-1)    # Flatning the inputs from tensors to vectors 
        #print(x.shape)
        logits = self.fc_net(x)        # Passing the conv layer to fully connected layer
        #print(x.shape)
        return logits
#------------------------------ BubLeNet --------------------------------------

#------------------------------ BubUNet --------------------------------------
class UNET(nn.Module): 
    def __init__(self, in_channels, first_out_channels, exit_channels, 
                 downhill, padding=0):
        super(UNET, self).__init__()
        self.encoder = Encoder(in_channels, first_out_channels, padding, downhill=downhill)
        self.decoder = Decoder(first_out_channels*(2**downhill), first_out_channels*(2**(downhill-1)),
                               exit_channels, padding=padding, uphill=downhill)
    
    def forward(self, x): 
        enc_out, routes = self.encoder(x) 
        out = self.decoder(enc_out, routes) 
        return out 

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=0):
        super(CNNBlock, self).__init__()

        self.seq_block = nn.Sequential( 
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True) 
        )
    
    def forward(self, x): 
        x = self.seq_block(x) 
        return x 

class CNNBlocks(nn.Module): 
    """
    Parameters:
    n_conv (int): creates a block of n_conv convolutions
    in_channels (int): number of in_channels of the first block's convolution
    out_channels (int): number of out_channels of the first block's convolution
    expand (bool) : if True after the first convolution of a block the number of channels doubles
    """
    def __init__(self, n_conv, 
                 in_channels, out_channels, padding): 
        super(CNNBlocks, self).__init__() 
        self.layers = nn.ModuleList() 
        
        for i in range(n_conv): 
            self.layers.append(CNNBlock(in_channels, out_channels, padding=padding)) 
            in_channels = out_channels 

    def forward(self, x): 
        for layer in self.layers: 
            x = layer(x) 
        return x 

class Encoder(nn.Module): 
    """
    Parameters:
    in_channels (int): number of in_channels of the first CNNBlocks
    out_channels (int): number of out_channels of the first CNNBlocks
    padding (int): padding applied in each convolution
    downhill (int): number times a CNNBlocks + MaxPool2D it's applied.
    """
    def __init__(self, in_channels, out_channels, 
                 padding, donwhill=4):
        super(Encoder, self).__init__() 
        self.enc_layers = nn.ModuleList() 
    
        for _ in range(donwhill): 
            self.enc_layers += [
                CNNBlock(2, in_channels, out_channels, padding),
                nn.MaxPool2d(2, 2) 
            ]

            in_channels = out_channels
            out_channels *= 2 
        
        self.enc_layers.append(CNNBlocks(2, in_channels, out_channels, padding)) 
    
    def forward(self, x): 
        route_connection = [] 
        for layer in self.enc_layers: 
            if isinstance(layer, CNNBlocks): 
                x = layer(x) 
                route_connection.append(x) 
            else: 
                x = layer(x) 
        return x, route_connection 
    
class Decoder(nn.Module): 
    """
    Parameters:
    in_channels (int): number of in_channels of the first ConvTranspose2d
    out_channels (int): number of out_channels of the first ConvTranspose2d
    padding (int): padding applied in each convolution
    uphill (int): number times a ConvTranspose2d + CNNBlocks it's applied.
    """
    def __init__(self, in_channels, out_channels, exit_channels,
                 padding, uphill=4): 
        super(Decoder, self).__init__() 
        self.exit_channels = exit_channels 
        self.layers = nn.ModuleList() 

        for i in range(uphill): 

            self.layers += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2), 
                CNNBlocks(2, in_channels, out_channels, padding) 
            ]
            in_channels //= 2
            out_channels //= 2
        
        self.layers.append(
            nn.Conv2d(in_channels, exit_channels, kernel_size=1, padding=padding) 
        )
    
    def forward(self, x, routes_connection): 
        routes_connection.pop(-1) 
        for layer in self.layers: 
            if isinstance(layer, CNNBlocks):
                routes_connection[-1] = F.center_crop(routes_connection[-1], x.shape[2])
                x = torch.cat([x, routes_connection.pop(-1)], dim=1) 
                x = layer(x) 
            else: 
                x = layer(x) 
        return x 


                