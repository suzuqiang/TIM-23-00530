from CF import *
#%%
def GetSomeLayer(LayerName,in_channels,out_channels=0):
    if LayerName == 'Blocks2DTorch':
        Layer = Blocks2DTorch(in_channels,out_channels)
    elif LayerName == 'ClasserTorch':
        Layer = ClasserTorch(in_channels,out_channels)
    elif LayerName == 'Simam2DPytorch':
        Layer = Simam2DPytorch(in_channels)
    elif LayerName == 'SeNet2DPytorch':
        Layer = SeNet2DPytorch(in_channels)
    elif LayerName == 'CBAM2DPytorch':
        Layer = CBAM2DPytorch(in_channels)
    elif LayerName == 'GCT2DPytorch':
        Layer = GCT2DPytorch(in_channels)
    elif LayerName == 'FcaNet2DPytorch':
        Layer = FcaNet2DPytorch(in_channels,out_channels)
    elif LayerName == 'FcaNetPlus2DPytorch':
        Layer = FcaNetPlus2DPytorch(in_channels,out_channels)
    elif LayerName == 'FcaNetPlusPlus2DPytorch':
        Layer = FcaNetPlusPlus2DPytorch(in_channels,out_channels)
    elif LayerName == 'ConvNet2DPytorch':
        Layer = ConvNet2DPytorch(in_channels,out_channels)
    LayerName =  Configure.FD+ 'Model'+LIST+'Initialization_Model_Parameter'+LIST+LayerName+'_'+str(in_channels)+'_'+str(out_channels)+'.pkl'
    
    if os.path.exists(LayerName):
        Layer.load_state_dict(torch.load(LayerName))
    else:
        torch.save(Layer.state_dict(),LayerName)
    return Layer
class Blocks2DTorch(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Blocks2DTorch, self).__init__(**kwargs)
        self.conv2d11 = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
        self.conv2d13 = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2d15 = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=5,stride=1,padding=2)
        self.conv2d17 = torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=7,stride=1,padding=3)
        self.mp = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU()
    def forward(self, inputs, training=None):
        return self.relu(self.bn(self.mp(self.conv2d11(inputs)+self.conv2d13(inputs)+self.conv2d15(inputs)+self.conv2d17(inputs))))
class ClasserTorch(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ClasserTorch, self).__init__( **kwargs)
        self.Blocks = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_channels, out_features = 1024),
            torch.nn.BatchNorm1d(num_features=1024),torch.nn.ReLU(),
            torch.nn.Linear(in_features = 1024, out_features = 128),
            torch.nn.BatchNorm1d(num_features=128),torch.nn.ReLU(),
            torch.nn.Linear(in_features = 128, out_features = out_channels),torch.nn.Softmax(1))
    def forward(self, inputs, training=None):
        out = self.Blocks(torch.flatten(inputs,1))
        return out
#%%Simam
class Simam2DPytorch(torch.nn.Module):
    def __init__(self, e_lambda = 1e-4):
        super(Simam2DPytorch, self).__init__()
        self.activaton = torch.nn.Sigmoid()
        self.e_lambda = e_lambda
    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)
#%%SeNet
class SeNet2DPytorch(torch.nn.Module):
    def __init__(self, channels):
        super(SeNet2DPytorch, self).__init__()
        self.channels = channels
        self.blocks = torch.nn.Sequential(
                      torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),
                      torch.nn.Linear(in_features=channels,out_features=channels//16),torch.nn.ReLU(),
                      torch.nn.Linear(in_features=channels//16,out_features=channels),torch.nn.Sigmoid())
    def forward(self, inputs):
        excitation = self.blocks(inputs).view((-1,self.channels,1,1))
        scale = inputs * excitation
        return scale
#%%CBAM 
class CBAM2DPytorch(torch.nn.Module):
    def __init__(self, channels):
        super(CBAM2DPytorch, self).__init__()
        self.channels = channels
        self.avg_out= torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
        self.max_out= torch.nn.Sequential(torch.nn.AdaptiveMaxPool2d(1),torch.nn.Flatten())
        self.blocks = torch.nn.Sequential(
                      torch.nn.Linear(in_features=channels,out_features=channels//16),torch.nn.ReLU(),
                      torch.nn.Linear(in_features=channels//16,out_features=channels),torch.nn.Sigmoid())
        self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=1,stride=1,padding=0),torch.nn.Sigmoid())
    def forward(self, inputs):
        channelattentionout = self.blocks(self.avg_out(inputs)) + self.blocks(self.max_out(inputs))
        channelattentionout = inputs*channelattentionout.view((-1,self.channels,1,1))
        spatialattentionout = self.conv(channelattentionout.mean(1,keepdim=True)+channelattentionout.max(1,keepdim=True)[0])
        spatialattentionout = channelattentionout*spatialattentionout
        out = spatialattentionout + inputs
        return out
#%%DCT
class LinearDCT2DPytorch(torch.nn.Module):
    def __init__(self, height, width, type, norm=None):
        super(LinearDCT2DPytorch, self).__init__()
        self.type,self.height,self.width,self.norm = type,height,width,norm
        self.linear_layer_1,self.linear_layer_2 = LinearDCT1DPytorch(width,self.type),LinearDCT1DPytorch(height,self.type)
    def forward(self,inputs):
        X1 = self.linear_layer_1(inputs)
        X2 = self.linear_layer_2(X1.transpose(-1, -2))
        return X2.transpose(-1, -2)
#%%FcaNet
class FcaNetPlus2DPytorch(torch.nn.Module):
    def __init__(self,channels,outchannels,freq_num=4):
        super(FcaNetPlus2DPytorch, self).__init__()
        self.channels,self.hight,self.weight,self.batch,self.freq_num = channels,0,0,0,freq_num
        self.mapper_x,self.mapper_y,self.matrix,self.Key = 0,0,0,0
        self.conv,self.linear = 0,torch.nn.Linear(freq_num**2,1,bias=False)
    def Something(self):
        if self.Key == 0:
            self.mapper_x,self.mapper_y=np.arange(0,self.hight,self.hight//self.freq_num),np.arange(0,self.weight,self.weight//self.freq_num)
            self.Key,self.matrix = 1,torch.zeros((self.freq_num**2,1,self.hight,self.weight))
            for i in range(len(self.mapper_x)):
                for j in range(len(self.mapper_y)):
                    for t_x in range(self.hight): 
                        for t_y in range(self.weight):
                            self.matrix[i*self.freq_num+j,0,t_x, t_y ]=self.build_filter(t_x,self.mapper_x[i],self.hight)*self.build_filter(t_y,self.mapper_y[j],self.weight)
            self.conv = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(self.hight,self.weight),stride=1,padding=0)
           
            self.matrix,self.conv = self.matrix.to("cuda:0"),self.conv.to("cuda:0")
        return self.matrix

    def build_filter(self, pos ,freq ,POS):
        result = np.cos(np.pi*freq*(pos+0.5)/POS)/np.sqrt(POS)
        if freq==0:
            return result
        else :
            return result*np.sqrt(2)
    def forward(self, inputs):
        self.batch,self.channels,self.hight,self.weight = inputs.size()
        weighted = (inputs.view(self.batch,1,self.channels,self.hight,self.weight)*self.Something()).view(-1,1,self.hight,self.weight)
        weighted = self.linear(self.conv(weighted).view(self.batch,self.channels,-1)).view(self.batch,self.channels,1,1)*inputs
        return weighted + inputs
class FcaNet2DPytorch(torch.nn.Module):
    def __init__(self,channels,outchannels,freq_num=4):
        super(FcaNet2DPytorch, self).__init__()
        self.channels,self.hight,self.weight,self.batch,self.freq_num = channels,0,0,0,freq_num
        self.mapper_x,self.mapper_y,self.matrix,self.Key = 0,0,0,0
        self.conv,self.linear1,self.linear2 = 0,torch.nn.Linear(freq_num**2,1,bias=False),torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(self.channels,self.channels//16),torch.nn.ReLU(),
        torch.nn.Linear(self.channels//16,self.channels),torch.nn.Sigmoid())
    def Something(self):
        if self.Key == 0:
            self.mapper_x,self.mapper_y=np.arange(0,self.hight,self.hight//self.freq_num),np.arange(0,self.weight,self.weight//self.freq_num)
            self.Key,self.matrix = 1,torch.zeros((self.freq_num**2,1,self.hight,self.weight))
            for i in range(len(self.mapper_x)):
                for j in range(len(self.mapper_y)):
                    for t_x in range(self.hight): 
                        for t_y in range(self.weight):
                            self.matrix[i*self.freq_num+j,0,t_x, t_y ]=self.build_filter(t_x,self.mapper_x[i],self.hight)*self.build_filter(t_y,self.mapper_y[j],self.weight)
            self.conv = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(self.hight,self.weight),stride=1,padding=0)
            self.matrix,self.conv = self.matrix.to("cuda:0"),self.conv.to("cuda:0")
        return self.matrix
    def build_filter(self, pos ,freq ,POS):
        result = np.cos(np.pi*freq*(pos+0.5)/POS)/np.sqrt(POS)
        if freq==0:
            return result
        else :
            return result*np.sqrt(2)
    def forward(self, inputs):
        self.batch,self.channels,self.hight,self.weight = inputs.size()
        weighted = (inputs.view(self.batch,1,self.channels,self.hight,self.weight)*self.Something()).view(-1,1,self.hight,self.weight)
        weighted = self.linear1(self.conv(weighted).view(self.batch,self.channels,-1))
        weighted = self.linear2(weighted).view(self.batch,self.channels,1,1)*inputs
        return weighted + inputs
class FcaNetPlusPlus2DPytorch(torch.nn.Module):
    def __init__(self,channels,outchannels,freq_num=4,epsilon=1e-5,mode='l1',after_relu=False):
        super(FcaNetPlusPlus2DPytorch, self).__init__()
        self.channels,self.hight,self.weight,self.batch,self.freq_num = channels,0,0,0,freq_num
        self.mapper_x,self.mapper_y,self.matrix,self.Key = 0,0,0,0
        self.conv,self.linear = 0,torch.nn.Linear(freq_num**2,1,bias=False)
        self.alpha = torch.nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon,self.mode,self.after_relu = epsilon,mode,after_relu
    def Something(self):
        if self.Key == 0:
            self.mapper_x,self.mapper_y=np.arange(0,self.hight,self.hight//self.freq_num),np.arange(0,self.weight,self.weight//self.freq_num)
            self.Key,self.matrix = 1,torch.zeros((self.freq_num**2,1,self.hight,self.weight))
            for i in range(len(self.mapper_x)):
                for j in range(len(self.mapper_y)):
                    for t_x in range(self.hight): 
                        for t_y in range(self.weight):
                            self.matrix[i*self.freq_num+j,0,t_x, t_y ]=self.build_filter(t_x,self.mapper_x[i],self.hight)*self.build_filter(t_y,self.mapper_y[j],self.weight)
            self.conv = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(self.hight,self.weight),stride=1,padding=0)
            
            self.matrix,self.conv = self.matrix.to("cuda:0"),self.conv.to("cuda:0")
        return self.matrix
    def build_filter(self, pos ,freq ,POS):
        result = np.cos(np.pi*freq*(pos+0.5)/POS)/np.sqrt(POS)
        if freq==0:
            return result
        else :
            return result*np.sqrt(2)
    def forward(self, inputs):
        self.batch,self.channels,self.hight,self.weight = inputs.size()
        weighted = (inputs.view(self.batch,1,self.channels,self.hight,self.weight)*self.Something())
        weighted = (weighted.pow(2).sum((3,4), keepdim=True) + self.epsilon).pow(0.5) * self.alpha +\
            self.conv(weighted.view(-1,1,self.hight,self.weight)).view(self.batch,-1,self.channels,1,1)
        weighted = (self.gamma/(weighted.pow(2).mean(dim=(1,2),keepdim=True)+self.epsilon).pow(0.5)).view(self.batch,self.channels,1,1)\
        +self.linear(weighted.view(self.batch,self.channels,-1)).view(self.batch,self.channels,1,1)
        return (weighted + 1.)*inputs
#%%
class GCT2DPytorch(torch.nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l1', after_relu=False):
        super(GCT2DPytorch, self).__init__()
        self.alpha = torch.nn.Parameter(torch.randn(1, num_channels, 1, 1))
        self.gamma = torch.nn.Parameter(torch.randn(1, num_channels, 1, 1))
        self.beta = torch.nn.Parameter(torch.randn(1, num_channels, 1, 1))
        self.epsilon,self.mode,self.after_relu = epsilon,mode,after_relu
    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2,3),keepdim=True)+self.epsilon).pow(0.5)*self.alpha
            norm = self.gamma/(embedding.pow(2).mean(dim=1,keepdim=True)+self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            _x = x if self.after_relu else torch.abs(x)
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma/(torch.abs(embedding).mean(dim=1,keepdim=True)+self.epsilon)
        gate = 1.+torch.tanh(embedding*norm+self.beta)
        return x*gate
class ConvNet2DPytorch(torch.nn.Module):
    def __init__(self,channels,num,epsilon=1e-5):
        super(ConvNet2DPytorch, self).__init__()
        self.channels,self.hight,self.weight,self.batch,self.num = channels,0,0,0,num
        self.alpha = torch.nn.Parameter(torch.randn(1, self.channels, 1, 1))
        self.gamma = torch.nn.Parameter(torch.randn(1, self.channels, 1, 1))
        self.beta = torch.nn.Parameter(torch.randn(1, self.channels, 1, 1))
        # self.linear2 = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(self.channels,self.channels//16),torch.nn.ReLU(),
        # torch.nn.Linear(self.channels//16,self.channels),torch.nn.Sigmoid())
    def init(self):
        self.conv = torch.nn.Conv2d(in_channels=1,out_channels=self.num,kernel_size=(self.hight,self.weight),stride=1,padding=0)
        self.linear1 = torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(self.num,self.num//8),torch.nn.ReLU(),
        torch.nn.Linear(self.num//8,1),torch.nn.Sigmoid())
        self.conv,self.linear1 = self.conv.to("cuda:0"),self.linear1.to("cuda:0")
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m,torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m,torch.nn.Linear) and m.bias is not None:
               torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.LayerNorm):
           torch.nn.init.constant_(m.bias, 0)
           torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m,torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, inputs):
        self.batch,self.channels,self.hight,self.weight = inputs.size()
        self.init()
        linear1out = self.linear1(self.conv(inputs.view(-1,1,self.hight,self.weight))).view(self.batch,self.channels,-1,1)
        norm = ((self.gamma/(torch.abs(linear1out).mean(dim=1,keepdim=True)+self.alpha))+self.beta)
        # linear2out = self.linear2(linear1out).view(self.batch,self.channels,1,1)
        weigth = 1.+torch.tanh(linear1out*norm)
        return  inputs*weigth
if __name__ == '__main__':
    inputs = torch.randn(128, 64, 32,32).cuda()
    net = FcaNet2DPytorch(64,64,4).cuda()
    print(net(inputs).shape)
    

