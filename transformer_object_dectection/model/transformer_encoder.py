import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable, Optional
from einops import einsum, rearrange
from typing import Optional, Tuple
from model.layers_ours import *
torch.set_float32_matmul_precision("high")
from matplotlib import pyplot as plt 
######################################################################################
####################################### MHSA #########################################
######################################################################################
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_model, n_heads, kv_heads, dropout, device):
        super().__init__()

        assert n_model % n_heads == 0

        self.n_model = n_model
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = n_model // n_heads
        self.kv_model = n_model // n_heads * kv_heads
        
        self.fc_q = nn.Linear(n_model, n_model).to(device)
        self.fc_k = nn.Linear(n_model, self.kv_model).to(device)
        self.fc_v = nn.Linear(n_model, self.kv_model).to(device)

        self.fc_o = nn.Linear(self.kv_model, n_model).to(device)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        #query = [batch size, query len, embedding size]
        #key = [batch size, key len, embedding size]
        #value = [batch size, value len, embedding size]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, embedding size]
        #K = [batch size, key len, embedding size]
        #V = [batch size, value len, embedding size]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        #print(Q.shape)
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        #energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        num_head_groups = self.n_heads // self.kv_heads
        if num_head_groups > 1:
            # Separate the query heads into 'num_head_groups' chunks, and fold the group
            # dimension into the batch dimension.  This allows us to compute the attention
            # for each head in parallel, then sum over all of the groups at the end.
            #Q = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
            Q = Q.view(batch_size, self.kv_heads, num_head_groups, -1, self.head_dim).permute(0,2,1,3,4)
            #print("einsum")
            #print(Q.shape)
            #print(K.shape)
            #torch.Size([12, 2, 4, 16, 64])
            #torch.Size([12, 4, 16, 64])
            energy = einsum(Q, K, "b g h n d, b h s d -> b h n s")


        #energy = [batch size, n heads, query len, key len]

        attention = torch.softmax(energy/self.scale, dim = -1)

        #attention = [batch size, n heads, query len, key len]

        #x = torch.matmul(self.dropout(attention), V)
        x = einsum(self.dropout(attention), V, "b h n s, b h s d -> b h n d")

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.kv_model)

        #x = [batch size, query len, embedding size]
        x = self.fc_o(x)

        #x = [batch size, query len, embedding size]

        return x, attention

######################################################################################
###################################### BitMGQA #######################################
######################################################################################
class BitMGQA(nn.Module):
    def __init__(self, n_model, n_heads, kv_heads, dropout, device):
        super().__init__()

        assert n_model % n_heads == 0
        # A = Q*K^T
        self.matmul1 = einsum2("b g h n d, b h s d -> b h n s")
        # attn = A*V
        self.matmul2 = einsum2("b h n s, b h s d -> b h n d")
        self.softmax = Softmax(dim=-1)
        
        self.n_model = n_model
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = n_model // n_heads
        self.kv_model = n_model // n_heads * kv_heads

        self.fc_q = BitLinear(n_model, n_model).to(device)
        self.fc_k = BitLinear(n_model, self.kv_model).to(device)
        self.fc_v = BitLinear(n_model, self.kv_model).to(device)
        
        self.fc_o = BitLinear(self.kv_model, n_model).to(device)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        eps = 1e-6
        denom = cam+eps*torch.sign(cam)
        denom = torch.where(denom == 0, torch.full_like(denom, eps), denom)
        cam = cam/(denom)
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def save_attn_gradients(self, attn_gradients):
        #print(attn_gradients)
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

        
    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        #query = [batch size, query len, embedding size]
        #key = [batch size, key len, embedding size]
        #value = [batch size, value len, embedding size]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, embedding size]
        #K = [batch size, key len, embedding size]
        #V = [batch size, value len, embedding size]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
        
        #energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        num_head_groups = self.n_heads // self.kv_heads
        if num_head_groups > 1:
            # Separate the query heads into 'num_head_groups' chunks, and fold the group
            # dimension into the batch dimension.  This allows us to compute the attention
            # for each head in parallel, then sum over all of the groups at the end.
            #Q = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
            Q = Q.view(batch_size, self.kv_heads, num_head_groups, -1, self.head_dim).permute(0,2,1,3,4)
            #print("einsum")
            #print(Q.shape)
            #print(K.shape)
            #torch.Size([12, 2, 4, 16, 64])
            #torch.Size([12, 4, 16, 64])
            #energy = einsum(Q, K, "b g h n d, b h s d -> b h n s")
            energy = self.matmul1([Q, K])


        #energy = [batch size, n heads, query len, key len]

        #attention = torch.softmax(energy/self.scale, dim = -1)
        attention = self.softmax(energy/self.scale)

        #attention = [batch size, n heads, query len, key len]
        #try:
        #    self.save_attn(attention)
        #    self._attn_hook = attention.register_hook(self.save_attn_gradients)
        #except:
        #    print("No gradients saved for Chefer method")
        #    pass
        
        #x = torch.matmul(self.dropout(attention), V)
        #x = einsum(self.dropout(attention), V, "b h n s, b h s d -> b h n d")
        x = self.matmul2([attention, V])

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.kv_model)

        #x = [batch size, query len, embedding size]
        x = self.fc_o(x)

        #x = [batch size, query len, embedding size]
        return x, attention
    def clean_hooks(self):
        if hasattr(self, "_attn_hook") and self._attn_hook is not None:
            self._attn_hook.remove()
            self._attn_hook = None

    def relprop(self, cam, **kwargs):     
        
        cam = self.fc_o.relprop(cam, **kwargs)

        #if cam.shape[1]==196:
        #    plt.figure(figsize=(12, 12))
        #    cam_test = cam[0].detach().cpu().permute(1, 0).reshape(-1, 14, 14)
        #    for i in range(64):
        #        plt.subplot(8, 8, i+1)
        #        plt.imshow(cam_test[i,:,:])
        #        if i+1>=cam_test.shape[0]:
        #            break
        
        #print("en attn relprop", cam.shape) # torch.Size([1, 196, 64])
                
        b, q_l, n = cam.shape
        cam = cam.reshape(b, q_l, self.kv_heads, -1)
        cam = cam.permute(0, 2, 1, 3)
        #cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.n_heads)
        #print("en attn rearrange", cam.shape) # [1, 8, 196, 8]            
        #en attn torch.Size([1, 197, 768])
        #en attn rearrange torch.Size([1, 12, 197, 64])
        
        # attn = A*V
        (cam1, cam_v)= self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        #if cam1.shape[2]==196:
        #    plt.figure(figsize=(12, 12))
        #    cam_test = cam1[0].detach().cpu().sum(dim=0).permute(1, 0).reshape(-1, 14, 14)
        #    for i in range(64):
        #        plt.subplot(8, 8, i+1)
        #        plt.imshow(cam_test[i,:,:])
        #        if i+1>=cam_test.shape[0]:
        #            break

        self.save_attn_cam(cam1)

        #cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)
        #print("en attn softmax", cam.shape) #[1, 4, 196, 16])
        
        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2
        #print("en attn matmul qkv", cam_q.shape, cam_k.shape, cam_v.shape) #[1, 2, 4, 196, 16]) ([1, 4, 196, 16]) ([1, 4, 196, 16]
        cam_q = cam_q.permute(0, 2, 1, 3, 4)
        cam_q = cam_q.reshape(b, self.n_heads, q_l, -1)
        #print("cam_q rearranged", cam_q.shape)
        #cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)
        cam_q = cam_q.permute(0, 2, 1, 3)
        cam_q = cam_q.reshape(b, -1, self.n_heads*self.head_dim)
        cam_k = cam_k.permute(0, 2, 1, 3)
        cam_k = cam_k.reshape(b, -1, self.kv_heads*self.head_dim)
        cam_v = cam_v.permute(0, 2, 1, 3)
        cam_v = cam_v.reshape(b, -1, self.kv_heads*self.head_dim)

        #print("en attn arranged qkv", cam_q.shape, cam_k.shape, cam_v.shape) #[1, 196, 128]) ([1, 196, 64]) ([1, 196, 64])
        cam_q = self.fc_q.relprop(cam_q, **kwargs)
        cam_k = self.fc_k.relprop(cam_k, **kwargs)
        cam_v = self.fc_v.relprop(cam_v, **kwargs)
        #print("cam_q ", cam_q.shape) #[1, 196, 128]
    
        #cam = (cam_q+cam_k+cam_v)/3
        
        return cam_q, cam_k, cam_v# self.qkv.relprop(cam_qkv, **kwargs)
######################################################################################
####################################### MLP ##########################################
######################################################################################
class MLP(nn.Module):
    def __init__(self, n_ff, n_model, dropout=0.5):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_model, n_ff),
            nn.GELU(),
            nn.LayerNorm(n_ff),
            nn.Dropout(dropout),
            nn.Linear(n_ff, n_model)
        )
    def forward(self, x):
        return self.ff(x)

######################################################################################
###################################### Bit FF ########################################
######################################################################################
class BitFeedForward(nn.Module):
    def __init__(self, n_ff, n_model, dropout=0.5):
        super().__init__()
        self.ff = nn.Sequential(
            BitLinear(n_model, n_ff, bias=True),
            GELU(),
            
            LayerNorm(n_ff),
            Dropout(dropout),
            BitLinear(n_ff, n_model, bias=True)
        )
    def forward(self, x):
        return self.ff(x)

    def relprop(self, cam, **kwargs):
        #cam = self.drop.relprop(cam, **kwargs)
        #print("bitff", cam.shape)
        cam = self.ff[-1].relprop(cam, **kwargs)

        #if cam.shape[1]==196:
        #    plt.figure(figsize=(12, 12))
        #    cam_test = cam[0].detach().cpu().permute(1, 0).reshape(-1, 14, 14)
        #    for i in range(36):
        #        plt.subplot(6, 6, i+1)
        #        plt.imshow(cam_test[i,:,:])
        #        plt.axis('off')
        #        if (i+1)>=cam_test.shape[0]:
        #            break
                    
        cam = self.ff[-2].relprop(cam, **kwargs)
        cam = self.ff[2].relprop(cam, **kwargs)
        cam = self.ff[1].relprop(cam, **kwargs)
        cam = self.ff[0].relprop(cam, **kwargs)
        #print("bit logrado", cam.shape)
        #cam = self.act.relprop(cam, **kwargs)
        #cam = self.fc1.relprop(cam, **kwargs)
        return cam    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size = 16, n_model = 512, in_channels = 1):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size, img_size[1]//patch_size) # grid size, size of new image according to patch size
        self.num_patches = self.grid_size[0] * self.grid_size[1] # grid size x grid size
        
        self.ln1 = LayerNorm(patch_size*patch_size*in_channels)
        self.ln2 = LayerNorm(n_model)
        self.proj = Linear(patch_size*patch_size*in_channels, n_model)
        self.activation = GELU()
    def forward(self, X):
        #input shape X : batch_size, num of channels image, height, weight
        #output shape : batch_size, num_patches, size_embedding
        X = rearrange( X, "b c (ht hp) (wt wp) -> b (ht wt) (hp wp c)", hp=self.patch_size , wp=self.patch_size)
        return self.ln2(self.activation(self.proj(self.ln1(X))))
    def relprop(self, cam, **kwargs):
        #print("block encoder ", cam.shape) #[1, 196, 128]    
        cam = self.ln1.relprop(cam, **kwargs)
            
        cam = self.proj.relprop(cam, **kwargs)
        #print("after linear en patch emb", cam.shape) #[1, 196, 256]
        
                    
        cam = self.activation.relprop(cam, **kwargs)
        cam = self.ln2.relprop(cam, **kwargs)
        
        #plt.figure(figsize=(12, 12))
        #print(cam.shape)
        #cam_test = cam[0].detach().cpu().permute(1, 0).reshape(-1, 14, 14)
        #for i in range(36):
        #    plt.subplot(6, 6, i+1)
        #    plt.imshow(cam_test[i,:,:])
        #    if (i+1)>=cam_test.shape[0]:
        #        break

        #if cam.shape[1]==784:
        #    plt.figure(figsize=(12, 12))
        #    cam_test = cam[0].detach().cpu().permute(1, 0).reshape(-1, 28, 28)
        #    for i in range(36):
        #        plt.subplot(6, 6, i+1)
        #        plt.imshow(cam_test[i+20,:,:])
            
        H, W = self.grid_size[0]*self.patch_size, self.grid_size[1]*self.patch_size
        # Recover original
        cam = rearrange(cam, "b (ht wt) (hp wp c) -> b c (ht hp) (wt wp)", 
                                ht=H//self.patch_size, wt=W//self.patch_size, 
                                hp=self.patch_size, wp=self.patch_size)
        #b = cam.shape[0]
        #cam = cam.permute(0, 2, 1)
        #print("permute patch emb", cam.shape)#torch.Size([1, 256, 196])
        #cam = cam.reshape(b, -1, self.grid_size[0]*self.patch_size, self.grid_size[1]*self.patch_size) 
        #print("reschape patch emb", cam.shape) #torch.Size([1, 64, 28, 28])
        #print("en el patch emb", cam.shape)
        return cam

class Block_encoder(nn.Module):
    def __init__(self, n_model, mult, num_heads, dropout, device, bitNet):
        super().__init__()
        self.ln1 = LayerNorm(n_model)
        if bitNet:
            kv_heads = num_heads//2
            #self.attention = BitMGQA(n_model, num_heads, kv_heads)
            #self.ffn =  BitFeedForward(n_model, n_model, mult, post_act_ln=True, dropout=dropout)
            self.attention = BitMGQA(n_model, num_heads, kv_heads, dropout, device)
            self.ffn =  BitFeedForward(n_model*mult, n_model, dropout)
        else:
            kv_heads = num_heads//2
            self.attention = MultiHeadAttentionLayer(n_model, num_heads, kv_heads, dropout, device)
            #self.attention = nn.MultiheadAttention(n_model, num_heads)
            self.ffn = MLP(n_model*mult, n_model, dropout)
        self.ln2 = LayerNorm(n_model)
        
        self.dropout = Dropout(dropout)
        self.att = None
        
        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()
        self.clone3 = Clone()
        
    def forward(self, X, mask=None):
        # X: batch_size, num_patches, size_embedding
        #X_norm = self.ln1(X)
        #att_output, att = self.attention(X_norm, X_norm, X_norm)
        #self.att = att
        #return torch.add( torch.add(X,self.dropout(att_output)), self.dropout(self.ffn(self.ln2(torch.add(X, self.dropout(att_output))))))
        #output batch_size, num_patches, size_embedding
        
        x1, x2 = self.clone1(X, 2)
        x2 = self.ln1(x2)
        x_q, x_k, x_v = self.clone3(x2, 3)
        X = self.add1([x1, self.attention(x_q, x_k, x_v)[0]])
        #X = self.add1([x1, self.attention(self.ln1(x2))])
        x3, x4 = self.clone2(X, 2)
        X = self.add2([x3, self.ffn(self.ln2(x4))])
        return X, x2
    def relprop(self, cam, **kwargs):
        #print("block encoder ", cam.shape) #[1, 784, 128]
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
                    
        cam2 = self.ffn.relprop(cam2, **kwargs)
                    
        cam2 = self.ln2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        #print("antes de attention todo bien")
        cam_q, cam_k, cam_v = self.attention.relprop(cam2, **kwargs)
        cam2 = self.clone3.relprop((cam_q, cam_k, cam_v), **kwargs)


        cam2 = self.ln1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)

        #if cam.shape[1]==196:
        #    print(self.attention.get_attn_gradients())
        #    plt.figure(figsize=(12, 12))
        #    cam_test = cam[0].detach().cpu().permute(1, 0).reshape(-1, 14, 14)
        #    for i in range(36):
        #        plt.subplot(6, 6, i+1)
        #        plt.imshow(cam_test[i,:,:])
        #        plt.axis('off')
        #        if (i+1)>=cam_test.shape[0]:
        #            break
                    
        return cam
    
class Encoder(nn.Module):
    def __init__(self, img_size, in_channels, n_model, mult, patch_size, num_heads=8, num_blks=1, blk_dropout=0.1, device='cpu', bitNet=True):
        super().__init__()
        self.size = img_size
        self.patch_size = patch_size
        
        #self.patch_embedding = PatchEmbedding(img_size, patch_size, n_model, in_channels=in_channels)
        #num_patches = self.patch_embedding.num_patches
        if patch_size==8:
            self.init_conv = nn.Sequential(Conv2d(in_channels, n_model//8, kernel_size=3, padding=1, bias=False),
                                          BatchNorm2d(n_model//8, momentum=0.9, eps=1e-5), GELU() )
        
            self.patch_embedding1 = PatchEmbedding(img_size, 2, n_model//4, in_channels=n_model//8)
            self.patch_embedding2 = PatchEmbedding((img_size[0]//2, img_size[1]//2), 2, n_model//2, in_channels=n_model//4)
            self.patch_embedding3 = PatchEmbedding((img_size[0]//4, img_size[1]//4), 2, n_model, in_channels=n_model//2)
            num_patches = 784
        else:
            self.patch_embedding = PatchEmbedding(img_size, patch_size, n_model, in_channels=in_channels)
            num_patches = self.patch_embedding.num_patches
        
        # Posicional embedding are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, n_model))
        
        self.dropout = Dropout(blk_dropout)
        #self.enc = Encoder(n_model, n_model*4, num_heads, num_blks, blk_dropout, device)
        self.ln = LayerNorm(n_model)

        self.blks = nn.ModuleList([Block_encoder(n_model, mult,
                                                num_heads, blk_dropout, device, bitNet) for _ in range(num_blks)])
        self.inp_grad = None
        self.add1 = Add()
        
    def forward(self, X):
        # X: batch_size, N channels, height, weight
        #X = self.patch_embedding(X)
        if self.patch_size==8:
            feats = []
            X = self.init_conv(X)
            feats.append(X)
            #X, feats = self.patch_embedding(X)
            X = self.patch_embedding1(X)
            
            B, _, n_model = X.size() 
            h, w = self.size[0]//2, self.size[1]//2
            #X = X[:,1:,:]#ignore first patch
            X = X.permute(0, 2, 1)
            X = X.contiguous().view(B, n_model, h, w)
            #print("x shape after emb1", X.shape)
        
            feats.append(X)
            X = self.patch_embedding2(X)
            
            B, _, n_model = X.size() 
            h, w = self.size[0]//4, self.size[1]//4
            #X = X[:,1:,:]#ignore first patch
            X = X.permute(0, 2, 1)
            X = X.contiguous().view(B, n_model, h, w)
            #print("x shape after emb2", X.shape)
            feats.append(X)
            X = self.patch_embedding3(X)
            
        else:
            X = self.patch_embedding(X)
        
        # X: batch_size, num_patches+1, size_embedding
        #X = self.dropout(X+self.pos_embedding)
        X = self.dropout(self.add1([X,self.pos_embedding]))

        for blk in self.blks:
            X, x2  = blk(X)
        X = self.ln(X)
        
        ############################################################################################
        ######################### Linear Projection of Flattened Patches ###########################
        ############################################################################################
        # reshape from (B, n_patch, n_model) to (B, n_model, h, w)
        B, n_patch, n_model = X.size() 
        h, w = self.size[0]//self.patch_size, self.size[1]//self.patch_size
        #X = X[:,1:,:]#ignore first patch
        X = X.permute(0, 2, 1)
        X = X.contiguous().view(B, n_model, h, w)
        
        if self.patch_size==8:
            return X, feats, x2
        else:
            return X, x2
        #return X, x2
    def relprop(self, cam, **kwargs):
        #[1, 128, 14, 14] #enc2
        #[1, 256, 14, 14]
        B, n_model, h, w = cam.shape
        cam = cam.permute(0, 2, 3, 1)
        cam = cam.reshape(B, h*w, n_model)
        #print("relprop en encoder", cam.shape) #[1, 196, 128]

        #if h==14:
        #    plt.figure(figsize=(12, 12))
        #    cam_test = cam[0].detach().cpu().permute(1, 0).reshape(128, 14, 14)
        #    for i in range(36):
        #        plt.subplot(6, 6, i+1)
        #        plt.imshow(cam_test[i+50,:,:])
            
        cam = self.ln.relprop(cam, **kwargs)
        #print("test relprop")
        #print(cam.shape)
                
        for blk in reversed(self.blks):
            cam = blk.relprop(cam, **kwargs)

        #if cam.shape[1]==784:
        #    plt.figure(figsize=(12, 12))
        #    #cam_test = cam[0].detach().cpu().permute(1, 0).reshape(-1, 14, 14)
        #    cam_test = cam[0].detach().cpu().permute(1, 0).reshape(-1, 28, 28)
        #    for i in range(36):
        #        plt.subplot(6, 6, i+1)
        #        plt.imshow(cam_test[i,:,:])
        #        plt.axis('off')
        #        if (i+1)>=cam_test.shape[0]:
        #            break
                    
            
        #(cam1, cam2) = self.add1.relprop(cam, **kwargs)
        #cam = self.dropout.relprop(cam, **kwargs)

        #cam2 = self.patch_embedding.relprop(cam, **kwargs) #torch.Size([1, 256, 14, 14])
        #print(cam2.shape)
        #print("estructura funciona", cam.shape) #[1, 196, 128]
        return cam 