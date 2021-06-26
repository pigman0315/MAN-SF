import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer

class gru(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.gru1 = nn.GRU(input_size = input_size, hidden_size=hidden_size, batch_first=True)
    def forward(self, inputs):
        full, last  = self.gru1(inputs)
        return full,last

class attn(nn.Module):
    def __init__(self,in_shape, out_shape ):
        super(attn, self).__init__()
        self.W1 = nn.Linear(in_shape, out_shape)
        self.W2 = nn.Linear(in_shape ,out_shape)
        self.V = nn.Linear(in_shape,1)
    def forward(self, full, last):
        score = self.V(torch.tanh(self.W1(last) + self.W2(full)))
        attention_weights = F.softmax(score, dim=1)
        context_vector = attention_weights * full
        context_vector =torch.sum(context_vector, dim=1) 
        return context_vector


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, alpha, stock_num):
        """Dense version of GAT."""
        super(GAT, self).__init__()

        ### Price Encoder
        self.grup = [gru(3,64) for _ in range(stock_num)]
        for i,gru_p in enumerate(self.grup):
            self.add_module('gru_p{}'.format(i), gru_p)

        self.attnp = [attn(64,64) for _ in range(stock_num)]
        for i,attn_p in enumerate(self.attnp):
            self.add_module('attn_p{}'.format(i), attn_p)


        ### Day-wise Tweet Encoder
        self.tweet_gru = [gru(512,64) for _ in range(stock_num)]
        for i,tweet_gru_ in enumerate(self.tweet_gru):
            self.add_module('tweet_gru{}'.format(i), tweet_gru_)

        self.attn_tweet = [attn(64,64) for _ in range(stock_num)]
        for i,attn_tweet_ in enumerate(self.attn_tweet):
            self.add_module('attn_tweet{}'.format(i), attn_tweet_)

         ### Sequential Tweet Encoder
        self.grut = [gru(64, 64) for _ in range(stock_num)]
        for i,gru_t in enumerate(self.grut):
            self.add_module('gru_t{}'.format(i), gru_t)

        self.attnt = [attn(64,64) for _ in range(stock_num)]
        for i,attnt_ in enumerate(self.attnt):
            self.add_module('attnt{}'.format(i), attnt_)


        ### Blending Multimodal
        self.bilinear = [nn.Bilinear(64,64,64) for _ in range(stock_num)]
        for i,bilinear_ in enumerate(self.bilinear):
            self.add_module('bilinear{}'.format(i), bilinear_)

        self.layer_normt = [nn.LayerNorm((1,64)) for _ in range(stock_num)]
        for i,layer_normt_ in enumerate(self.layer_normt):
            self.add_module('layer_normt{}'.format(i), layer_normt_)

        self.layer_normp = [nn.LayerNorm((1,64)) for _ in range(stock_num)]
        for i,layer_normp_ in enumerate(self.layer_normp):
            self.add_module('layer_normp{}'.format(i), layer_normp_)

        self.linear_x = [nn.Linear(64,nclass) for _ in range(stock_num)]
        for i,linear_x_ in enumerate(self.linear_x):
            self.add_module('linear_x{}'.format(i), linear_x_)

        ### GAT
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, text_input, price_input, adj):
        li = []
        num_tw = text_input.size(2) # number of tweet  = NUM_TWEET = 8
        num_d = price_input.size(1) # number of day = 5
        pr_ft = price_input.size(2) # number of price feature = 3
        num_stocks = price_input.size(0) # number of company

        ### Price Encoder & SMI Encoder
        for i in range(price_input.size(0)): # For Company
            # Price Encoder
            x = self.grup[i](price_input[i,:,:].reshape((1,num_d,pr_ft)))
            x = self.attnp[i](*x).reshape((1,64))
            han_li1 = []
            for j in range(text_input.size(1)): # For Day
                # Day-wise Tweet Encoder
                y = self.tweet_gru[i](text_input[i,j,:,:].reshape(1,num_tw,512)) # 512: USE embed size
                y = self.attn_tweet[i](*y).reshape((1,64))
                han_li1.append(y)
            news_vector = torch.Tensor((1,num_d,64))
            news_vector = torch.cat(han_li1)
            # Sequential Tweet Encoder
            text = self.grut[i](news_vector.reshape(1,num_d,64))
            text = self.attnt[i](*text).reshape((1,64))
            combined = torch.tanh(self.bilinear[i](text, x).reshape((1,64)))
            li.append(combined.reshape(1,64))
        ft_vec = torch.Tensor((num_stocks,64))
        ft_vec = torch.cat(li)
        out_1 = torch.tanh(self.linear_x[i](ft_vec))

        ### GAT
        x = F.dropout(ft_vec, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x+out_1
    """
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    """


# class SpGAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Sparse version of GAT."""
#         super(SpGAT, self).__init__()
#         self.dropout = dropout

#         self.attentions = [SpGraphAttentionLayer(nfeat, 
#                                                  nhid, 
#                                                  dropout=dropout, 
#                                                  alpha=alpha, 
#                                                  concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = SpGraphAttentionLayer(nhid * nheads, 
#                                              nclass, 
#                                              dropout=dropout, 
#                                              alpha=alpha, 
#                                              concat=False)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)

