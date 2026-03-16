#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:04:12 2022

@author: aglabassi
"""

import torch
import torch_geometric

#对 MILP 二部图中的变量特征、约束特征、边系数、全局 bounds 做“问题感知的归一化”，让 GNN 看到的数值尺度稳定、可比较、可泛化。
# def normalize_graph(constraint_features, 
#                     edge_index,
#                     edge_attr,
#                     variable_features,
#                     bounds,
#                     depth,
#                     bound_normalizor = 1000):
    
    
#     #SMART
#     # 获取目标函数系数的绝对值的最大值，用该值归一化所有变量的目标函数系数
#     obj_norm = torch.max(torch.abs(variable_features[:,2]), axis=0)[0].item()
#     var_max_bounds = torch.max(torch.abs(variable_features[:,:2]), axis=1, keepdim=True)[0]  
    
#     var_max_bounds.add_(var_max_bounds == 0)
    
#     var_normalizor = var_max_bounds[edge_index[0]]
#     cons_normalizor = constraint_features[edge_index[1], 0:1]
#     normalizor = var_normalizor/(cons_normalizor + (cons_normalizor == 0))
    
#     variable_features[:,2].div_(obj_norm)
#     variable_features[:,:2].div_(var_max_bounds)
#     constraint_features[:,0].div_(constraint_features[:,0] + (constraint_features[:,0] == 0) )
#     edge_attr.mul_(normalizor)
#     bounds.div_(bound_normalizor)
    
#     return (constraint_features, edge_index, edge_attr, variable_features, bounds, depth)

def normalize_graph(constraint_features, edge_index, edge_attr, variable_features, bounds, depth, bound_normalizor=1000):
    
    # ================== 异常数值深度诊断 ==================
    
    if variable_features.numel() > 0:
        max_val = torch.max(variable_features)
        if max_val > 1e7:
            print("-----------------normalize_graph之前-----------------")
            # 找到最大值的扁平化索引
            flat_idx = torch.argmax(variable_features)
            # 转换为 (行, 列) 坐标
            row = flat_idx // variable_features.shape[1]
            col = flat_idx % variable_features.shape[1]
            
            print(f"\n" + "!"*30)
            print(f"CRITICAL VALUE DETECTED!")
            print(f"最大数值: {max_val.item():.2e}")
            print(f"出现位置: 行 {row.item()}, 列 {col.item()}")
            print(f"该行完整特征: \n{variable_features[row]}")
            
            # 针对性提示
            col_map = {0:"sol", 1:"lb", 2:"ub", 5:"obj", 9:"is_sol_inf"}
            col_name = col_map.get(col.item(), "其他列")
            print(f"初步诊断: 大数出现在 [{col_name}]，请检查相关数据源。")
            print("!"*30 + "\n")
    # ====================================================

    # 1. 目标系数归一化 (对应第5列: obj_coeff)
    obj_vals = variable_features[:, 5]
    obj_norm = torch.max(torch.abs(obj_vals)).item()
    if obj_norm < 1e-9: obj_norm = 1.0 # 验证保护：防止全0导致NaN
    obj_vals.div_(obj_norm)
    
    # 2. 变量 Bound 归一化 (对应第1, 2列: lb, ub)
    # 我们顺便把第0列的 sol 也带上，因为解的大小随 Bound 缩放
    var_max_bounds = torch.max(torch.abs(variable_features[:, 1:3]), axis=1, keepdim=True)[0]
    var_max_bounds = torch.clamp(var_max_bounds, min=1e-9) # 优雅替代 .add_(==0)
    
    variable_features[:, 0:3].div_(var_max_bounds) # 同时缩放 sol, lb, ub
    
    # 3. 边属性缩放 (edge_attr)
    # var_normalizor = var_max_bounds[edge_index[0]]
    # cons_normalizor = torch.abs(constraint_features[edge_index[1], 0:1])
    # normalizor = var_normalizor / (cons_normalizor + (cons_normalizor == 0))
    # edge_attr.mul_(normalizor)
    if edge_attr.numel() > 0:
        edge_norm = torch.max(torch.abs(edge_attr)).item()
        if edge_norm > 1e-9:
            edge_attr.div_(edge_norm)
        else:
            edge_attr.div_(1e-9)
    else:
        print("Warning: 子图没有边，跳过边归一化。")
    
    # 4. 约束归一化
    # constraint_features[:, 0].div_(torch.abs(constraint_features[:, 0]) + (constraint_features[:, 0] == 0))
    cons_max_vals = torch.max(torch.abs(constraint_features[:, 0:2]), dim=1, keepdim=True)[0]
    cons_max_vals = torch.clamp(cons_max_vals, min=1e-9)
    # 重点：同时缩放第 0 列和第 1 列
    constraint_features[:, 0:2].div_(cons_max_vals)
    
    # 5. 步数归一化
    bounds.div_(bound_normalizor)

    # 打印每一列的极大值，看看归一化到底有没有把该压的压下来
    # for i in range(variable_features.shape[1]):
    #     col_max = variable_features[:, i].max().item()
    #     print(f"变量特征第 {i} 列最大值: {col_max:.2e}")
    
    
    return (constraint_features, edge_index, edge_attr, variable_features, bounds, depth)

#function definition
# https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
def process(policy, data_loader, loss_fct, device, optimizer=None, normalize=True):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for idx,batch in enumerate(data_loader):
            
            
            batch = batch.to(device)
            if normalize:
                #IN place operations
                (batch.constraint_features_s,
                 batch.edge_index_s, 
                 batch.edge_attr_s,
                 batch.variable_features_s,
                 batch.bounds_s,
                 batch.depth_s)  =  normalize_graph(batch.constraint_features_s,  batch.edge_index_s, batch.edge_attr_s,
                                                    batch.variable_features_s, batch.bounds_s,  batch.depth_s)
                
                (batch.constraint_features_t,
                 batch.edge_index_t, 
                 batch.edge_attr_t,
                 batch.variable_features_t,
                 batch.bounds_t,
                 batch.depth_t)  =  normalize_graph(batch.constraint_features_t,  batch.edge_index_t, batch.edge_attr_t,
                                                    batch.variable_features_t, batch.bounds_t,  batch.depth_t)
                                                    
        
            y_true = 0.5*batch.y + 0.5 #0,1 label from -1,1 label
            y_proba = policy(batch)
            y_pred = torch.round(y_proba)
            
            # Compute the usual cross-entropy classification loss
            #loss_fct.weight = torch.exp((1+torch.abs(batch.depth_s - batch.depth_t)) / 
                            #(torch.min(torch.vstack((batch.depth_s,  batch.depth_t)), axis=0)[0]))

            l = loss_fct(y_proba, y_true)
            loss_value = l.item()
            if optimizer is not None:
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            
            accuracy = (y_pred == y_true).float().mean().item()

            mean_loss += loss_value * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs
            #print(y_proba.item(), y_true.item())

    mean_loss /= (n_samples_processed + ( n_samples_processed == 0))
    mean_acc /= (n_samples_processed  + ( n_samples_processed == 0))
    return mean_loss, mean_acc


def process_ranknet(policy, X, y, loss_fct, device, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0
    n_samples_processed = 0
    X.to(device)

    with torch.set_grad_enabled(optimizer is not None):
        for idx,x in enumerate(X):
            yi = y[idx].to(device)
            y_true = 0.5*yi + 0.5 #0,1 label from -1,1 label
            y_proba = policy(x[:20].to(device), x[20:].to(device))
            y_pred = torch.round(y_proba)
            
            # Compute the usual cross-entropy classification loss
            #loss_fct.weight = torch.exp((1+torch.abs(batch.depth_s - batch.depth_t)) / 
                            #(torch.min(torch.vstack((batch.depth_s,  batch.depth_t)), axis=0)[0]))
            #print(y_proba)
            l = loss_fct(y_proba, y_true)
            #print(l)
            loss_value = l.item()
            if optimizer is not None:
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            
            accuracy = (y_pred == y_true).float().mean().item()

            mean_loss += loss_value
            mean_acc += accuracy 
            n_samples_processed += 1
            #print(y_proba.item(), y_true.item())

    mean_loss /= (n_samples_processed + ( n_samples_processed == 0))
    mean_acc /= (n_samples_processed  + ( n_samples_processed == 0))
    return mean_loss, mean_acc

