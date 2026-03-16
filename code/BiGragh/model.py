#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 20:44:58 2021

@author: abdel
from https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb with some modifications
"""

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GraphConv
import torch.nn as nn
# from data_type import BipartiteGraphPairData



class RankNet(torch.nn.Module):

    def __init__(self):
        super(RankNet, self).__init__()

        self.linear1 = torch.nn.Linear(20, 50)
        self.activation = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(50, 1)
        
    def forward_node(self, n):
        x = self.linear1(n)
        x = self.activation(x)
        x = self.linear2(x)
        return x
        

    def forward(self, n0,n1):
        s0,s1 = self.forward_node(n0), self.forward_node(n1)
   
        return torch.sigmoid(-s0 + s1)
    
    
    

class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_size = emb_size = 32 #uniform node feature embedding dim
        
        hidden_dim1 = 8
        hidden_dim2 = 4
        hidden_dim3 = 4
        
        # static data
        cons_nfeats = 6
        edge_nfeats = 1
        var_nfeats = 10

        output_dim = 15
        

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            # torch.nn.LayerNorm(edge_nfeats),
            torch.nn.Linear(edge_nfeats, 1)
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
        )
        
        self.bounds_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(2),
            torch.nn.Linear(2,2),
            torch.nn.ReLU(),
        )
        
        self.convs_v2c = torch.nn.ModuleList([
            GraphConv((self.emb_size, self.emb_size), hidden_dim1),
            GraphConv((hidden_dim1, hidden_dim1), hidden_dim2),
            GraphConv((hidden_dim2, hidden_dim2), hidden_dim3)
        ])
        
        # 【修复 2】定义反向路径的卷积（或者保持对称性）
        self.convs_c2v = torch.nn.ModuleList([
            GraphConv((self.emb_size, self.emb_size), hidden_dim1),
            GraphConv((hidden_dim1, hidden_dim1), hidden_dim2),
            GraphConv((hidden_dim2, hidden_dim2), hidden_dim3)
        ])


        #double check
 
        # self.convs = []
        
        # self.conv1 = GraphConv((emb_size, emb_size), hidden_dim1 )
        # self.conv2 = GraphConv((hidden_dim1, hidden_dim1), hidden_dim2 )
        # self.conv3 = GraphConv((hidden_dim2, hidden_dim2), hidden_dim3 )
        
        # self.convs = [ self.conv1, self.conv2, self.conv3 ]

        # 4. 投影层 (Projection Layer)
        # 输入维度 = 变量池化(hidden_dim3) + 约束池化(hidden_dim3) + 边界特征(8)
        combined_input_dim = hidden_dim3 + hidden_dim3 + 2
        self.feature_projection = nn.Sequential(
            nn.Linear(combined_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # out_size = hidden_dim3 if len(self.convs)==3 else emb_size
        
        # self.final_mlp = torch.nn.Sequential( 
        #                     torch.nn.Linear(2*out_size+2, 1, bias=False),
        #                     torch.nn.Sigmoid()
        #                     )
           
       
    def forward_graph(self, constraint_features, edge_indices, edge_features, 
                       variable_features, bbounds, constraint_batch=None, variable_batch=None):

        '''  
        constraint_features: [3744, 4]      # 总共3744个约束节点，每个4维特征
        variable_features: [6048, 6]        # 总共6048个变量节点，每个6维特征  
        edge_indices: [2, 14672]            # 14672条边（变量->约束）
        edge_features: [14672, 1]           # 每条边1维特征
        bbounds: [16, 2]                    # 16个图的边界特征
        constraint_batch: [3744]            # 约束节点的批次索引
        variable_batch: [6048]              # 变量节点的批次索引
        '''
        # 1. 物理连续化洗白 (消除所有 CUBLAS 报错的隐患)
        variable_features = variable_features.contiguous()
        constraint_features = constraint_features.contiguous()
        edge_features = edge_features.contiguous()
        edge_indices = edge_indices.contiguous()
        bbounds = bbounds.contiguous()
        
        # if constraint_batch is not None:
        #     constraint_batch = constraint_batch.contiguous().long()
        # if variable_batch is not None:
        #     variable_batch = variable_batch.contiguous().long()

        # 1. 嵌入层投影
        variable_features = self.var_embedding(variable_features) #[6048, 32]
        constraint_features = self.cons_embedding(constraint_features) #[3774,32]
        edge_features = self.edge_embedding(edge_features) #[14672,1]
        bbounds = self.bounds_embedding(bbounds) # 此时 bbounds 维度为 [B, 2] = [16, 2]
        
        # print("111111111111111111111111111111111")
        # print(f"DEBUG: constraint_features shape: {constraint_features.shape}")
        # print(f"DEBUG: variable_features shape: {variable_features.shape}")
        # print(f"DEBUG: edge_indices shape: {edge_indices.shape}")
        # print(f"DEBUG: edge_features shape: {edge_features.shape}")
        # print(f"DEBUG: bbounds shape: {bbounds.shape}")   

        # num_vars = variable_features.size(0)  # 应该是 6048
        # num_cons = constraint_features.size(0) # 应该是 3744
        # max_var_idx = edge_indices[0].max() # Var -> Cons，假设 0 维是 Var
        # max_cons_idx = edge_indices[1].max()

        # print(f"--- Edge Index Integrity Check ---")
        # print(f"Total Vars: {num_vars}, Max Var Index in Edges: {max_var_idx}")
        # print(f"Total Cons: {num_cons}, Max Cons Index in Edges: {max_cons_idx}")
        
        # 2. 准备反向边索引（用于双向消息传递）
        ## edge_indices: [2, 14672] 第一行是变量索引，第二行是约束索引
        edge_indices_reversed = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        # edge_indices_reversed: [2, 14672] 第一行是约束索引，第二行是变量索引

        for conv_v2c, conv_c2v in zip(self.convs_v2c, self.convs_c2v):
            # Var -> Cons      constraint_features_next: [3744, 16]
            constraint_features_next = F.relu(conv_v2c((variable_features, constraint_features), 
                                                edge_indices,
                                                edge_weight=edge_features))
            # Cons -> Var   variable_features: [6048, 16]
            variable_features = F.relu(conv_c2v((constraint_features, variable_features), 
                                              edge_indices_reversed,
                                              edge_weight=edge_features))

            constraint_features = constraint_features_next # [3744, 16]
        # 3. GNN 卷积循环
        # for conv in self.convs:
            
        #     #Var to cons
        #     constraint_features_next = F.relu(conv((variable_features, constraint_features), 
        #                                       edge_indices,
        #                                       edge_weight=edge_features,
        #                                       size=(variable_features.size(0), constraint_features.size(0))))
            
        #     #cons to var 
        #     variable_features = F.relu(conv((constraint_features, variable_features), 
        #                               edge_indices_reversed,
        #                               edge_weight=edge_features,
        #                               size=(constraint_features.size(0), variable_features.size(0))))
            
            # constraint_features = constraint_features_next
        # print("222222222222222222222222222222")
        # print(f"DEBUG: constraint_features shape: {constraint_features.shape}")
        # print(f"DEBUG: variable_features shape: {variable_features.shape}")
        # print(f"DEBUG: edge_indices shape: {edge_indices.shape}")
        # print(f"DEBUG: edge_features shape: {edge_features.shape}")
        # print(f"DEBUG: bbounds shape: {bbounds.shape}")   

        # DEBUG: constraint_features shape: torch.Size([3744, 16])
        # DEBUG: variable_features shape: torch.Size([6048, 16])
        # DEBUG: edge_indices shape: torch.Size([2, 14672])
        # DEBUG: edge_features shape: torch.Size([14672, 1])
        # DEBUG: bbounds shape: torch.Size([16, 2])

        # num_vars = variable_features.size(0)  # 应该是 6048
        # num_cons = constraint_features.size(0) # 应该是 3744
        # max_var_idx = edge_indices[0].max() # Var -> Cons，假设 0 维是 Var
        # max_cons_idx = edge_indices[1].max()

        # print(f"--- Edge Index Integrity Check ---")
        # print(f"Total Vars: {num_vars}, Max Var Index in Edges: {max_var_idx}")
        # print(f"Total Cons: {num_cons}, Max Cons Index in Edges: {max_cons_idx}")
        
        # 4. 全局池化
        # 4. 全局池化 (修正版)
        if constraint_batch is not None and variable_batch is not None:
            # constraint_batch = constraint_batch.to(constraint_features.device).long().contiguous()
            # variable_batch = variable_batch.to(variable_features.device).long().contiguous()
            # batch_size = bbounds.size(0)
            # 使用 global_mean_pool，参数顺序是 (特征, batch索引)
            # 结果形状自动为 [Batch_Size, Hidden]
            ## 输出: [16, 16]  # 16个图，每个图16维特征
            constraint_avg = torch_geometric.nn.global_mean_pool(constraint_features, constraint_batch)
            # 输入: constraint_features [3744, 16], constraint_batch [3744]
            # 操作: 按batch索引分组，每组取平均
            # 应该输出: [16, 16]  # 16个图，每个图16维特征
            # 实际输出却是 [32765, 16]
            variable_avg = torch_geometric.nn.global_mean_pool(variable_features, variable_batch)

            
            
        else:
            # 推理阶段：保持不变
            constraint_avg = torch.mean(constraint_features, dim=0, keepdim=True) # [1, 16]
            variable_avg = torch.mean(variable_features, dim=0, keepdim=True) # [1, 16]
        
        # 5. 维度对齐安全检查
        # 确保 bbounds 至少是 2 维的 [B, 2]，以便与 [B, Hidden] 拼接
        if bbounds.dim() == 1:
            bbounds = bbounds.unsqueeze(0)

        # print("333333333333333333333333333333333")
        # print(f"DEBUG - var_avg: {variable_avg.shape}")
        # print(f"DEBUG - cons_avg: {constraint_avg.shape}")
        # print(f"DEBUG - bbounds: {bbounds.shape}")
        combined = torch.cat((variable_avg, constraint_avg, bbounds), dim=1)   # [16, 34]
        # print(f"DEBUG: combined shape: {combined.shape}")
        
        output = self.feature_projection(combined) # # [16, 34] → [16, 32]
        return  output
        #返回了一个标量
        # return torch.linalg.norm(variable_avg, dim=1) + torch.linalg.norm(constraint_avg, dim=1) + torch.linalg.norm(bbounds, dim=1)
    

    



 
