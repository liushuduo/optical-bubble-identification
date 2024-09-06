from torch import nn

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
