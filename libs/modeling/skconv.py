import torch
import torch.nn as nn


class SKConv(nn.Module):
    def __init__(self, dim, M, r=2, act_layer=nn.GELU):
        """ Constructor
        Args:
            dim: input channel dimensionality.
            M: the number of branchs.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        self.dim = dim
        self.channel = dim
        self.d = self.channel // r
        self.M = M
        self.proj = nn.Linear(dim, dim)

        self.act_layer = act_layer()
        self.gap = nn.AdaptiveAvgPool2d((None,1))
        self.fc1 = nn.Linear(dim, self.d)
        self.fc2 = nn.Linear(self.d, self.M * self.channel)
        self.softmax = nn.Softmax(dim=1)
        self.proj_head = nn.Linear(self.channel, dim)
        self.conv1=nn.Conv2d(dim,self.channel,1)

    def forward(self, input_feats):
        group=input_feats
        bs,num_head,T,head_channels=input_feats.shape
        input_feats_head = input_feats.permute(0,1,3,2)
        input_feats=torch.sum(input_feats_head,dim=1)
        bs,C,T=input_feats.shape
        input_groups=group.reshape(bs,self.M,self.channel,T)
        feats = self.proj(input_feats.permute(0,2,1))   
        feats_proj = feats.permute(0,2,1).reshape(bs,self.dim,T) 
        feats = self.act_layer(feats)
        feats=feats.permute(0,2,1).reshape(bs,self.dim,T)
        feats_S = self.gap(feats) 
        feats_Z = self.fc1(feats_S.squeeze()) 
        feats_Z = self.act_layer(feats_Z) 
        attention_vectors = self.fc2(feats_Z) 
        attention_vectors = attention_vectors.view(bs, self.M, self.channel, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(input_groups * attention_vectors, dim=1) 
        feats_V = self.proj_head(feats_V.reshape(bs,self.channel,T).permute(0,2,1)) 
        feats_V = feats_V.permute(0,2,1)
        output = feats_proj + feats_V
        return output


class ESK(nn.Module):
    def __init__(self,dim,M,r=2, act_layer=nn.GELU):
        super(ESK,self).__init__()
        
        # *****************Channel******************** #
        self.dim = dim
        self.channel = dim
        self.d = self.channel // r
        self.M = M
        self.proj = nn.Linear(dim, dim)
        self.act_layer = act_layer()
        self.gap = nn.AdaptiveAvgPool2d((None,1))
        self.fc1 = nn.Linear(dim, self.d)
        self.fc2 = nn.Linear(self.d, self.M * self.channel)
        self.softmax = nn.Softmax(dim=1)
        self.proj_head = nn.Linear(self.channel, dim)
        self.conv1=nn.Conv1d(self.channel,self.M,1)
        # *****************Spatial******************** #
        self.relu=act_layer()
        self.softmaxS=nn.Softmax(dim=1)
        self.relu2=nn.ReLU()

    def forward(self,input_feats):
        
        group=input_feats
        bs,num_head,T,head_channels=input_feats.shape
        input_feats_head = input_feats.permute(0,1,3,2) 
        input_feats=torch.sum(input_feats_head,dim=1) 
        bs,C,T=input_feats.shape
        input_groups=group.reshape(bs,self.M,self.channel,T)
        feats = self.proj(input_feats.permute(0,2,1))  
        feats_proj = feats.permute(0,2,1).reshape(bs,self.dim,T) 
        feats = self.act_layer(feats)
        feats=feats.permute(0,2,1).reshape(bs,self.dim,T)
        
        spatial_feats=feats
        spatial_feats=self.relu(spatial_feats)
        spatial_feats_attention=self.conv1(spatial_feats) 
        spatial_feats_attention=self.relu2(spatial_feats_attention).view(bs,self.M,T,1)
        attention_vectors_S = self.softmaxS(spatial_feats_attention).permute(0,1,3,2) 
        feats_Spat = torch.sum(input_groups * attention_vectors_S, dim=1) 
        feats_S = self.gap(feats) 
        feats_Z = self.fc1(feats_S.squeeze()) 
        feats_Z = self.act_layer(feats_Z) 
        attention_vectors = self.fc2(feats_Z) 
        attention_vectors = attention_vectors.view(bs, self.M, self.channel, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(input_groups * attention_vectors, dim=1)
        feats_V = self.proj_head(feats_V.reshape(bs,self.channel,T).permute(0,2,1))
        feats_V = feats_V.permute(0,2,1)
        output = feats_V+ feats_Spat
        return output


if __name__=="__main__":
    # out
    B,num_head,T,head_channels=4,4,16,512

    input_ferats=torch.randn(B,num_head,T,head_channels)

    sk=SKConv(512,4)
    y=sk(input_ferats)
    # print(y.shape)
