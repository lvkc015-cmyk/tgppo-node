#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:16:39 2021

@author: abdel

Contains utilities to save and load comparaison behavioural data


"""

import os
import imp
import torch
import numpy as np
import re
import time
import gc

# def load_src(name, fpath):
#      return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

# load_src("data_type", "../learning/data_type.py" )

# from data_type import BipartiteGraphPairData


class CompFeaturizerSVM():
    def __init__(self, model,save_dir=None, instance_name=None):
        self.instance_name = instance_name
        self.save_dir = save_dir
        self.m = model
        
    def save_comp(self, model, node1, node2, comp_res, comp_id):
        
        f1,f2 = self.get_features(node1), self.get_features(node2)
        
        
        file_path = os.path.join(self.save_dir, f"{self.instance_name}_{comp_id}.csv")
        file = open(file_path, 'a')
        
        np.savetxt(file, f1, delimiter=',')
        np.savetxt(file, f2, delimiter=',')
        file.write(str(comp_res))
        file.close()
        
        return self
    
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        return self

    def get_features(self, node):
        
        model = self.m
        
        f = []
        feat = node.getHeHeaumeEisnerFeatures(model, model.getDepth()+1 )
        
        
        for k in ['vals', 'depth', 'maxdepth' ]:
            if k == 'vals':
                
                for i in range(1,19):
                    try:
                        f.append(feat[k][i])
                    except:
                        f.append(0)
                    
            else:
                f.append(feat[k])

        return f
    




class CompFeaturizer():
    
    def __init__(self, save_dir=None, instance_name=None):
        self.instance_name = instance_name
        self.save_dir = save_dir
        
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
        return self

    
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.LP_feature_recorder = LP_feature_recorder
        return self
        
    
    def save_comp(self, model, node1, node2, comp_res, comp_id):
        
        torch_geometric_data = self.get_torch_geometric_data(model, node1, node2, comp_res)
        file_path = os.path.join(self.save_dir, f"{self.instance_name}_{comp_id}.pt")
        torch.save(torch_geometric_data, file_path, _use_new_zipfile_serialization=False)
        
        return self
    
    def get_torch_geometric_data(self, model, node1, node2, comp_res=0):
        
        triplet = self.get_triplet_tensors(model, node1, node2, comp_res)
        
        return BipartiteGraphPairData(*triplet[0], *triplet[1], triplet[2])
    
    
    
    
    def get_graph_for_inf(self,model, node):
        
        gpu_gpu = time.time()
        
        self.LP_feature_recorder.record_sub_milp_graph(model, node)
        graphidx2graphdata = self.LP_feature_recorder.recorded_light
        all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
        all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
        
        g_idx = node.getNumber()
        
        var_attributes, cons_block_idxs = graphidx2graphdata[g_idx]
        
    
        g_data = self._get_graph_data(var_attributes, cons_block_idxs, all_conss_blocks, all_conss_blocks_features)
        
        variable_features = g_data[0]
        constraint_features = g_data[1]
        edge_indices = g_data[2]
        edge_features = g_data[3]
        
        lb, ub = node.getLowerbound(), node.getEstimate()
        depth = node.getDepth()
        
        if model.getObjectiveSense() == 'maximize':
            lb,ub = ub,lb
            
        g = (constraint_features,
              edge_indices, 
              edge_features, 
              variable_features, 
              torch.tensor([[lb, -1*ub]], device=self.LP_feature_recorder.device).float(),
              torch.tensor([depth], device=self.LP_feature_recorder.device).float()
              )
        
        gpu_gpu = (time.time() - gpu_gpu)
            
        return gpu_gpu, g
        
        
        
        
        
        
        
        
    
    
    
    def get_triplet_tensors(self, model, node1, node2, comp_res=0):
                
        self.LP_feature_recorder.record_sub_milp_graph(model, node1)
        self.LP_feature_recorder.record_sub_milp_graph(model, node2)
        graphidx2graphdata = self.LP_feature_recorder.recorded_light
        all_conss_blocks = self.LP_feature_recorder.all_conss_blocks
        all_conss_blocks_features = self.LP_feature_recorder.all_conss_blocks_features
        
        g0_idx, g1_idx, comp_res = node1.getNumber(), node2.getNumber(), comp_res
        
        var_attributes0, cons_block_idxs0 = graphidx2graphdata[g0_idx]
        var_attributes1, cons_block_idxs1 = graphidx2graphdata[g1_idx]
        
        g_data = self._get_graph_pair_data(var_attributes0, 
                                           var_attributes1, 
                                                         
                                           cons_block_idxs0, 
                                           cons_block_idxs1, 

                                           all_conss_blocks, 
                                           all_conss_blocks_features, 
                                           comp_res)
        
        bounds0 = [node1.getLowerbound(), node1.getEstimate()]
        bounds1 = [node2.getLowerbound(), node2.getEstimate()]
        
        if model.getObjectiveSense() == 'maximize':
            bounds0[1], bounds0[0] = bounds0
            bounds1[1], bounds1[0] = bounds1
            
        return self._to_triplet_tensors(g_data, node1.getDepth(), node2.getDepth(), bounds0, bounds1, self.LP_feature_recorder.device)
    
       
    
    def _get_graph_pair_data(self, var_attributes0, var_attributes1, cons_block_idxs0, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features, comp_res ):
        
        g1 = self._get_graph_data(var_attributes0, cons_block_idxs0, all_conss_blocks, all_conss_blocks_features)
        g2 = self._get_graph_data(var_attributes1, cons_block_idxs1, all_conss_blocks, all_conss_blocks_features)
     
        return list(zip(g1,g2)) + [comp_res]
    
    def _get_graph_data(self, var_attributes, cons_block_idxs, all_conss_blocks, all_conss_blocks_features):
        
        
        adjacency_matrixes = map(all_conss_blocks.__getitem__, cons_block_idxs)
        
        cons_attributes_blocks = map(all_conss_blocks_features.__getitem__, cons_block_idxs)
        
        #TO DO ACCELERATE HSTACK VSTACK
        # adjacency_matrix = torch.hstack(tuple(adjacency_matrixes))
        # cons_attributes = torch.vstack(tuple(cons_attributes_blocks))
        adjacency_matrix = tuple(adjacency_matrixes)[0]
        cons_attributes = tuple(cons_attributes_blocks)[0]
        
        edge_idxs = adjacency_matrix._indices()
        edge_features =  adjacency_matrix._values().unsqueeze(1)
            
        
        return var_attributes, cons_attributes, edge_idxs, edge_features
        
    
    def _to_triplet_tensors(self, g_data, depth0, depth1, bounds0, bounds1, device ):
        
        variable_features = g_data[0]
        constraint_features = g_data[1]
        edge_indices = g_data[2]
        edge_features = g_data[3]
        y = g_data[4]
        lb0, ub0 = bounds0
        lb1, ub1 = bounds1
        
        g1 = (constraint_features[0],
              edge_indices[0], 
              edge_features[0], 
              variable_features[0], 
              torch.tensor([[lb0, -1*ub0]], device=device).float(),
              torch.tensor([depth0], device=device).float()
              )
        g2 = (constraint_features[1], 
              edge_indices[1], 
              edge_features[1], 
              variable_features[1], 
              torch.tensor([[lb1, -1*ub1]], device=device).float(),
              torch.tensor([depth1], device=device).float()
              )
        
        
        
        return (g1,g2,y)
    
        
        
        

    


from collections import OrderedDict
class LPFeatureRecorder():
    #model：SCIP 模型;  device：GNN 使用的设备（CPU / GPU）
    def __init__(self, model, device):
        
        varrs = model.getVars() #所有变量
        original_conss = model.getConss() #原始约束（根节点）
        
        self.model = model
        
        self.n0 = len(varrs) #记录 原始变量个数（根问题规模）。
        
        #保存变量和约束列表。
        self.varrs = varrs
        self.original_conss = original_conss
        
        #初始化缓存结构
        self.recorded = dict()
        #轻量缓存
        self.recorded_light = dict()
        #存储 约束块结构
        self.all_conss_blocks = []
        self.all_conss_blocks_features = []
        #用于存目标函数相关的邻接结构（这里尚未用到）。
        self.obj_adjacency  = None
        
        self.device = device
        self.khop_fail_count = 0
        self.total_call_count = 0
        
        
        #INITIALISATION OF A,b,c into a graph 根节点图的初始化
        #开始计时
        self.init_time = time.time()
        #变量字符串 → 连续索引, 后续构图时统一变量编号
        # self.var2idx = dict([ (str_var, idx) for idx, var in enumerate(self.varrs) for str_var in [str(var)]  ])
        self.var2idx = {v.name: i for i, v in enumerate(self.varrs)}
        self.name_to_var = {v.name: v for v in self.varrs}
        for v in self.varrs:
            if not v.name.startswith("t_"):
                self.name_to_var[f"t_{v.name}"] = v

        #把 根 LP（原始问题）转成一个 图表示
        # root_graph = self.get_root_graph(model, device='cpu')
        #记录根图构建时间
        self.init_time = (time.time() - self.init_time)
        
        
        self.init_cpu_gpu_time = time.time()
        #变量特征移到 GPU
        # root_graph.var_attributes = root_graph.var_attributes.to(device)
        #所有约束块结构，统一拷贝到 GPU
        for idx, _ in  enumerate(self.all_conss_blocks_features): #1 single loop
            self.all_conss_blocks[idx] = self.all_conss_blocks[idx].to(device)
            self.all_conss_blocks_features[idx] = self.all_conss_blocks_features[idx].to(device)
        
        #记录 CPU→GPU 开销。
        self.init_cpu_gpu_time = (time.time() - self.init_cpu_gpu_time)
       
        #缓存根节点（编号 = 1）
        # self.recorded[1] = root_graph
        #缓存轻量版本。
        # self.recorded_light[1] = (root_graph.var_attributes, root_graph.cons_block_idxs)

        self.var_to_conss = {}      # {var_name: [cons_objects]}
        self.cons_to_vars = {}      # {cons_name: [var_names]}
        self.cons_to_coeffs = {}    # {cons_name: {var_name: value}}
        self.cons_to_nz_count = {}  # {cons_name: int}
        for cons in self.original_conss:
            # 只处理线性约束，因为 GNN 通常基于线性结构
            if not cons.isLinear():
                continue
            c_name = cons.name
            # 核心：只在这里调用一次昂贵的 getValsLinear
            var_coeffs = model.getValsLinear(cons)

            clean_coeffs = {}
            v_names = []
            for v, coeff in var_coeffs.items():
                v_name = v.name if hasattr(v, 'name') else str(v)
                clean_coeffs[v_name] = coeff
                v_names.append(v_name)
                
                # 填充反向索引
                if v_name not in self.var_to_conss:
                    self.var_to_conss[v_name] = []
                self.var_to_conss[v_name].append(cons)

            self.cons_to_vars[c_name] = v_names
            self.cons_to_coeffs[c_name] = clean_coeffs
            self.cons_to_nz_count[c_name] = len(v_names)


            # self.cons_to_vars[c_name] = list(var_coeffs.keys())
            # self.cons_to_coeffs[c_name] = var_coeffs
            # self.cons_to_nz_count[c_name] = len(var_coeffs)

            # for v in var_coeffs.keys():
            #     if isinstance(v, str):
            #         v_name = v
            #     else:
            #         v_name = v.name

            #     if v_name not in self.var_to_conss:
            #         self.var_to_conss[v_name] = []
            #     self.var_to_conss[v_name].append(cons)
                
        # self.name_to_var = {}
        # for v in self.varrs: # varrs 是 model.getVars() 拿到的
        #     self.name_to_var[str(v)] = v
        #     # 兼容变换后的前缀
        #     self.name_to_var[f"t_{str(v)}"] = v
        # print(f"--- 验证 A: 初始索引表样例 ---")
        # if self.var_to_conss:
        #     sample_key = list(self.var_to_conss.keys())[0]
        #     print(f"索引表中的变量名样例: '{sample_key}'")
        

       
    def clear(self):
        self.recorded.clear()
        self.recorded_light.clear()
        self.all_conss_blocks.clear()
        self.all_conss_blocks_features.clear()
        gc.collect()
        
    #获取某个节点的图
    #sub_milp: SCIP 的某个 B&B 节点
    def get_graph(self, model, sub_milp):
        
        #获取节点编号。
        sub_milp_number = sub_milp.getNumber()
        #如果已经构造过,直接返回缓存
        if sub_milp_number in self.recorded:
            return self.recorded[ sub_milp_number]
        else:
            #先构造, 再返回
            self.record_sub_milp_graph(model, sub_milp)
            return self.recorded[ sub_milp_number ]
        
    #构造子节点图
    # def record_sub_milp_graph(self, model, sub_milp):
        
    #     #确保只构造一次
    #     if sub_milp.getNumber() not in self.recorded:
            
    #         #获取父节点
    #         parent = sub_milp.getParent()
    #         # 如果是根节点,直接构造根图
    #         if parent == None: #Root
    #             graph = self.get_root_graph(model)
                
    #         else:
    #             #如果是子节点,复制父节点的图
    #             graph = self.get_graph(model, parent).copy()
    #             #self._add_conss_to_graph(graph, model, sub_milp.getAddedConss())
    #             #更新变量上下界,不改图结构,只改 变量特征
    #             self._change_branched_bounds(graph, sub_milp)
                
    #         #self._add_scip_obj_cons(model, sub_milp, graph)
    #         self.recorded[sub_milp.getNumber()] = graph
    #         self.recorded_light[sub_milp.getNumber()] = (graph.var_attributes, 
    #                                                      graph.cons_block_idxs)



    def record_sub_milp_graph(self, model, sub_milp, cands=None, k_hops=2):
       
        if sub_milp.getNumber() in self.recorded:
            return

        # 如果极端情况下 cands 为 None (例如 SCIP 初始状态)，我们主动获取一次种子
        if cands is None:
            print("候选变量为空")
            cands, _, _ = model.getCandsState(self.var_dim, self.branch_count)
            # 如果依然拿不到，截取前 100 个变量作为兜底种子（防止程序崩溃）
            if not cands:
                cands = model.getVars()[:100]
            else:
                cands = cands[:400] # 强制执行 400 种子截断

        # 此时产生的 graph 对象的维度是 [N_local]，索引是 [0...N_local-1]
        graph = self._extract_khop_manual(model, sub_milp, cands, k_hop=k_hops)

        # === 深度审计开始 ===
        # print(f"\n" + "="*50)
        # print(f"DEBUG: 正在审计 Sub-MILP #{sub_milp.getNumber()} 的图数据")
        # print(f"节点规模: 变量 {graph.n_vars}")
        # # 1. 变量特征审计 (9维)
        # if hasattr(graph, 'var_attributes') and graph.var_attributes is not None:
        #     v_feats = graph.var_attributes
        #     print(f"变量特征形状: {v_feats.shape}")
        #     # 重点排查：找出那个 2.62e08 到底在第几列
        #     for col in range(v_feats.shape[1]):
        #         c_max = torch.max(v_feats[:, col]).item()
        #         c_min = torch.min(v_feats[:, col]).item()
        #         c_nan = torch.isnan(v_feats[:, col]).any().item()
        #         print(f"  维度 {col}: [min={c_min:.2e}, max={c_max:.2e}] {'<-- !! NaN !!' if c_nan else ''}")
        # else:
        #     print("警告: 变量特征 (var_attributes) 为空！")

        # # 2. 约束特征审计
        # if hasattr(graph, 'cons_attributes') and graph.cons_attributes is not None:
        #     c_feats = graph.cons_attributes
        #     print(f"约束特征形状: {c_feats.shape}")
        #     if c_feats.shape[0] > 0:
        #         print(f"  约束第一列极值: [min={torch.min(c_feats[:,0]):.2e}, max={torch.max(c_feats[:,0]):.2e}]")
        #     else:
        #         print("  !!! 关键报错点确认：约束数量为 0，这会导致 GNN 崩溃 !!!")
        
        # # 3. 边信息审计
        # if hasattr(graph, 'local_edge_index'):
        #     e_idx = graph.local_edge_index
        #     e_attr = graph.local_edge_attr
        #     print(f"边索引形状: {e_idx.shape}, 边属性形状: {e_attr.shape}")
        #     if e_idx.numel() > 0:
        #         print(f"  最大变量索引: {torch.max(e_idx[0]):.0f}, 最大约束索引: {torch.max(e_idx[1]):.0f}")
        #         print(f"  边权重极值: [min={torch.min(e_attr):.2e}, max={torch.max(e_attr):.2e}]")
        #     else:
        #         print("  !!! 警告：边数量为 0 (孤立子图) !!!")
        
        # print("="*50 + "\n")
        # === 深度审计结束 ===

        self.recorded[sub_milp.getNumber()] = graph
        # self.recorded_light[sub_milp.getNumber()] = (graph.var_attributes, graph.cons_block_idxs)

    

    def _extract_khop_manual(self, model, sub_milp, cands, k_hop=2, max_edges=30000):
        def clean_name(v_name):
            return v_name[2:] if v_name.startswith("t_") else v_name

        
        dev = self.device
        
        # 存放选中的约束
        selected_conss_dict = {} 
        # 当前变量字典:{全局索引:变量对象}
        current_vars_dict = {v.getIndex(): v for v in cands}
        total_edges = 0
        limit_reached = False #是否截断

        # SCIP 无穷大定义
        SCIP_INF = model.infinity()

        # for 循环:从一批起始变量出发，按“变量 → 约束 → 变量 → …”的方式，做一个有边数上限的 K-hop 子图扩张
        # 得到一组 “被选中的约束集合 selected_conss_dict,并且满足边数不超过 max_edges。
        for i in range(k_hop):
            #创建一个空字典，用于存放新发现的约束
            new_conss_dict = {}
            for v_idx, var in current_vars_dict.items():

                v_name = var.name

                #用你自己预建的索引表找相关约束
                related_conss = self.var_to_conss.get(clean_name(v_name), [])
                # 尝试 B: 去掉 t_ 前缀匹配 (Y...)
                if not related_conss and v_name.startswith("t_"):
                    related_conss = self.var_to_conss.get(clean_name(v_name), [])

                #把新约束加入本 hop
                for cons in related_conss:
                    if cons.name not in selected_conss_dict:
                        new_conss_dict[cons.name] = cons
            
            # 计算新增边的数量  
            # current_hop_edges = sum(self.cons_to_nz_count.get(c.name, 0) for c in new_conss_dict.values())
            # current_hop_edges = sum(len(model.getValsLinear(c)) for c in new_conss_dict.values())
            
            # 如果这一步加进来会超过阈值，则停止扩张
            # if total_edges + current_hop_edges > max_edges:
            #     print(f"Warning: K-Hop truncated at step {i} to avoid OOM (Total Edges: {total_edges})")
            #     break

            # # 更新已选约束
            # selected_conss_dict.update(new_conss_dict)
            # total_edges += current_hop_edges


            # 我们给每个新发现的约束计算一个得分，得分越低（越接近 0）说明越紧，越优先
            scored_conss = []
            for cons in new_conss_dict.values():
                #把当前 LP 解代入约束，算出来的数值
                activity = model.getActivity(cons)
                #获取约束条件左侧的值
                lhs = model.getLhs(cons)
                #获取约束条件右侧的值
                rhs = model.getRhs(cons)
                
                # 计算到最近边界的距离 (Slack)
                dist_lhs = abs(activity - lhs) if lhs > -SCIP_INF else float('inf')
                dist_rhs = abs(activity - rhs) if rhs < SCIP_INF else float('inf')
                score = min(dist_lhs, dist_rhs)
                scored_conss.append((score, cons))
            # 按得分（紧迫度）从小到大排序
            scored_conss.sort(key=lambda x: x[0])

            # 按顺序塞入约束，直到边数预算耗尽
            for _, cons in scored_conss:
                nz = self.cons_to_nz_count.get(cons.name, 0)
                if total_edges + nz <= max_edges:
                    if cons.name not in selected_conss_dict:
                        selected_conss_dict[cons.name] = cons
                        total_edges += nz
                else:
                    print(f"Warning: Hard truncation by Activity Score at step {i}. Edges: {total_edges}")
                    limit_reached = True
                    break

            # 跳出“当前所在的那一层循环”
            if limit_reached:
                break

            # current_hop_conss = list(new_conss_dict.values())
            # limit_reached = False
            
            # for cons in current_hop_conss:
            #     nz = self.cons_to_nz_count.get(cons.name, 0)
                
            #     # 策略：i==0 是保底逻辑，必须让种子变量有约束连；i>0 则受 max_edges 限制
            #     if i == 0 or (total_edges + nz <= max_edges):
            #         if cons.name not in selected_conss_dict:
            #             selected_conss_dict[cons.name] = cons
            #             total_edges += nz
            #     else:
            #         print(f"Warning: K-Hop truncated at step {i} (Total Edges: {total_edges})")
            #         limit_reached = True
            #         break # 停止当前 Hop 的继续添加
            
            # if limit_reached:
            #     break # 停止后续 Hop 的扩张

            

            # 如果还没到最后一步，则寻找 约束 -> 变量 (偶数步)
            if i < k_hop - 1:
                next_vars_dict = {}
                for cons_name, cons in selected_conss_dict.items():
                    # 只有在 current_hop 真正被加入的约束才发散下一跳
                    if cons_name in new_conss_dict:
                        for v_name in self.cons_to_vars.get(cons_name, []):
                            v_obj = self.name_to_var.get(clean_name(v_name))
                            if v_obj: next_vars_dict[v_obj.getIndex()] = v_obj
                        
                current_vars_dict = next_vars_dict


        # --- 步骤 2: 同步补齐变量 (严格限制，防止变量节点撑爆) ---
        #创建一个“有顺序的字典”，用来按插入顺序存变量 把候选变量作为“核心变量”
        final_vars_dict = OrderedDict()
        for v in cands:
            final_vars_dict[v.getIndex()] = v

        #补齐所有与已选约束相关的变量,从约束反向补变量（保证连通）
        #从“已选中的约束”出发，把和这些约束有关的变量，全部收集进 final_vars_dict，而且不重复、按顺序保存。
        for cons in selected_conss_dict.values():
            for v_name in self.cons_to_vars.get(cons.name, []):
                v_obj = self.name_to_var.get(clean_name(v_name))
                if v_obj and v_obj.getIndex() not in final_vars_dict:
                    # 变量节点上限设定为 30000，防止特征矩阵过大
                    if len(final_vars_dict) < 8000:
                        final_vars_dict[v_obj.getIndex()] = v_obj
                    else:
                        break
                        
                

        # --- 步骤 3: 构建局部连续映射并填充特征 ---
        selected_vars = list(final_vars_dict.values())
        selected_conss = list(selected_conss_dict.values())
        
        graph = BipartiteGraphStatic0(len(selected_vars), dev)

        var_map, cons_map = self._add_selected_entities_to_graph(
            graph, model, selected_vars, selected_conss, dev
        )
         #把当前节点上的 变量 bound 变化 更新进图特征
        self._change_branched_bounds_local(graph, sub_milp, var_map)


        # --- 步骤 4: 构建边索引 (物理硬上限) ---
        #  构建边索引 (使用局部连续 ID) ---
        # 这一步非常关键：GNN 需要的是 [0, N_local] 之间的索引
        # 把“变量节点”和“约束节点”按线性约束里的系数连起来
        edge_indices = []
        edge_values = []
        curr_e = 0
        for c in selected_conss:
            c_local_idx = cons_map[c.name]
            # 获取该约束涉及的所有变量及其系数
            # 建议预存: self.cons_to_coeffs = {c.name: {v_name: val, ...}}
            coeffs = self.cons_to_coeffs.get(c.name, {})
            for v_name, val in coeffs.items():
                v_obj = self.name_to_var.get(clean_name(v_name))
                if v_obj and v_obj.getIndex() in var_map:
                    v_local_idx = var_map[v_obj.getIndex()]
                    edge_indices.append([v_local_idx, c_local_idx])
                    edge_values.append(val)
                    if curr_e >= max_edges: break
            if curr_e >= max_edges: break


        # --- 调试断点：如果还是没边，打印出名字看看 ---
        if not edge_indices and selected_conss:
            sample_c = selected_conss[0].name
            sample_vs = list(self.cons_to_coeffs.get(sample_c, {}).keys())[:3]
            print(f"DEBUG: 匹配失败! 约束 {sample_c} 下的变量名样例: {sample_vs}")
            print(f"DEBUG: name_to_var 里的样例 Key: {list(self.name_to_var.keys())[:3]}")

        # --- 【新增】步骤 5: 封装给 Brancher 使用的数据属性 ---
        # 即使没有边，也要保证 Tensor 形状正确 [2, 0]
        if edge_indices:
            graph.local_edge_index = torch.tensor(edge_indices, device=dev).t().contiguous()
            raw_edge_attr = torch.tensor(edge_values, dtype=torch.float32, device=dev).unsqueeze(1)
            graph.local_edge_attr = torch.sign(raw_edge_attr) * torch.log1p(torch.abs(raw_edge_attr))
            # graph.local_edge_attr = torch.tensor(edge_values, device=dev).unsqueeze(1).float()
        else:
            graph.local_edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
            graph.local_edge_attr = torch.empty((0, 1), dtype=torch.float32, device=dev)

        # 将约束特征也挂载到 graph 上
        # graph.local_cons_feats = graph.cons_attributes
        # graph.local_var_feats = graph.var_attributes

        return graph


    # def _extract_khop_manual(self, model, sub_milp, cands, k_hop=2, max_edges=100000):
    #     def clean_name(v_name):
    #         return v_name[2:] if v_name.startswith("t_") else v_name

    #     dev = self.device
    #     selected_conss_dict = {} #已选中的约束
    #     current_vars_dict = {v.getIndex(): v for v in cands}
    #     total_edges = 0
    #     limit_reached = False

    #     # --- 步骤 1: 带有硬截断的 K-hop 约束提取 ---
    #     for i in range(k_hop):
    #         new_conss_dict = {}
    #         # 找出当前变量参与了哪些约束
    #         for v_idx, var in current_vars_dict.items():
    #             v_name = var.name
    #             # 查找相关约束
    #             related_conss = self.var_to_conss.get(clean_name(v_name), [])
    #             if not related_conss and v_name.startswith("t_"):
    #                 related_conss = self.var_to_conss.get(clean_name(v_name)[2:], [])

    #             for cons in related_conss:
    #                 if cons.name not in selected_conss_dict:
    #                     new_conss_dict[cons.name] = cons
            
    #         # 当前跳收集的约束
    #         current_hop_conss = list(new_conss_dict.values())
            
    #         for cons in current_hop_conss:
    #             #该约束的非零系数数量
    #             nz = self.cons_to_nz_count.get(cons.name, 0)
                
    #             # 硬性截断逻辑：只要还没到上限，就往里塞
    #             if total_edges + nz <= max_edges:
    #                 selected_conss_dict[cons.name] = cons
    #                 total_edges += nz
    #             else:
    #                 # 实现你的逻辑：如果是第一跳，允许塞入最后一个使其“满载”
    #                 if i == 0 and total_edges < max_edges:
    #                     selected_conss_dict[cons.name] = cons
    #                     total_edges += nz # 此时 total_edges 会略微超过 max_edges，但随后立即 break
                    
    #                 print(f"Warning: K-Hop FORCE truncated at step {i} (Edges reached: {total_edges})")
    #                 limit_reached = True
    #                 break 
            
    #         if limit_reached:
    #             break

    #         # 准备下一跳的变量
    #         if i < k_hop - 1:
    #             next_vars_dict = {}
    #             for cons in new_conss_dict.values():
    #                 # 注意：只从本跳真正被选中的约束里找下一跳变量
    #                 if cons.name in selected_conss_dict:
    #                     for v_name in self.cons_to_vars.get(cons.name, []):
    #                         v_obj = self.name_to_var.get(clean_name(v_name))
    #                         if v_obj: next_vars_dict[v_obj.getIndex()] = v_obj
    #             current_vars_dict = next_vars_dict

    #     # --- 步骤 2: 同步补齐变量 (只补齐与选中约束相关的) ---
    #     final_vars_dict = OrderedDict()
    #     for v in cands:
    #         final_vars_dict[v.getIndex()] = v

    #     for cons_name, cons in selected_conss_dict.items():
    #         for v_name in self.cons_to_vars.get(cons_name, []):
    #             v_obj = self.name_to_var.get(clean_name(v_name))
    #             if v_obj and v_obj.getIndex() not in final_vars_dict:
    #                 # 变量节点也给一个安全上限，防止超大约束拉入数十万个变量
    #                 if len(final_vars_dict) < 50000: 
    #                     final_vars_dict[v_obj.getIndex()] = v_obj
    #                 else:
    #                     break

    #     # --- 步骤 3: 构建局部映射 ---
    #     selected_vars = list(final_vars_dict.values())
    #     selected_conss = list(selected_conss_dict.values())
        
    #     graph = BipartiteGraphStatic0(len(selected_vars), dev)

    #     var_map, cons_map = self._add_selected_entities_to_graph(
    #         graph, model, selected_vars, selected_conss, dev
    #     )
    #     self._change_branched_bounds_local(graph, sub_milp, var_map)

    #     # --- 步骤 4: 构建边索引 (带物理上限保护) ---
    #     edge_indices = []
    #     edge_values = []
    #     physical_edge_count = 0
        
    #     for c in selected_conss:
    #         if physical_edge_count >= max_edges:
    #             break
                
    #         c_local_idx = cons_map[c.name]
    #         coeffs = self.cons_to_coeffs.get(c.name, {})
            
    #         for v_name, val in coeffs.items():
    #             v_obj = self.name_to_var.get(clean_name(v_name))
    #             if v_obj and v_obj.getIndex() in var_map:
    #                 v_local_idx = var_map[v_obj.getIndex()]
                    
    #                 edge_indices.append([v_local_idx, c_local_idx])
    #                 edge_values.append(val)
                    
    #                 physical_edge_count += 1
    #                 if physical_edge_count >= max_edges:
    #                     break

    #     # --- 步骤 5: 封装 Tensor ---
    #     if edge_indices:
    #         # 使用更高效的方式转换 Tensor
    #         edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long, device=dev).t().contiguous()
    #         edge_attr_tensor = torch.tensor(edge_values, dtype=torch.float32, device=dev).unsqueeze(1)
            
    #         graph.local_edge_index = edge_index_tensor
    #         graph.local_edge_attr = edge_attr_tensor
    #     else:
    #         graph.local_edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
    #         graph.local_edge_attr = torch.empty((0, 1), dtype=torch.float32, device=dev)

    #     return graph

    # 把“选中的变量 / 约束”映射到一个紧凑的局部编号空间，并只为它们计算特征，填进图里。
    def _add_selected_entities_to_graph(self, graph, model, selected_vars, selected_conss, device):
       
        dev = device if device else self.device
        
        # 1. 建立变量的局部映射,重新从0开始编号,比如
        # var_global_to_local = {
        #     17(全局索引): 0(现在的编号),
        #     203: 1,
        #     5: 2
        # }
        var_global_to_local = {v.getIndex(): i for i, v in enumerate(selected_vars)}
        # 2. 建立约束的局部映射,全局名字:索引编号
        cons_global_to_local = {c.name: i for i, c in enumerate(selected_conss)}

        # 创建变量特征矩阵
        graph.var_attributes = torch.zeros(len(selected_vars), graph.d0, device=dev).float()
        # 创建一个局部的约束特征矩阵
        graph.cons_attributes = torch.zeros(len(selected_conss), graph.d1, device=dev).float()
        
        # 3. 仅为选中的变量计算特征 (存储在局部连续的位置)
        # 此时 graph.var_attributes 是一个形状为 (len(selected_vars), feat_dim) 的 Tensor
        for v in selected_vars:
            local_idx = var_global_to_local[v.getIndex()]
            graph.var_attributes[local_idx] = self._get_feature_var(model, v, dev)
            
        # 4. 仅为选中的约束计算特征
        for c in selected_conss:
            local_idx = cons_global_to_local[c.name]
            graph.cons_attributes[local_idx] = self._get_feature_cons(model, c, dev)
        
        # ================== 初始化阶段特征审计 ==================
        if graph.var_attributes.numel() > 0:
            max_val = torch.max(graph.var_attributes)
            if max_val > 1e7:
                print("------------------------_add_selected_entities_to_graph")
                flat_idx = torch.argmax(graph.var_attributes)
                row = flat_idx // graph.var_attributes.shape[1]
                col = flat_idx % graph.var_attributes.shape[1]
                
                # 获取该变量的原始 SCIP 变量对象（用于打印名字）
                v_obj = selected_vars[row.item()]
                
                print(f"\n" + "X"*20 + " 初始化阶段发现大数 " + "X"*20)
                print(f"节点编号: {model.getCurrentNode().getNumber()}")
                print(f"大数数值: {max_val.item():.2e}")
                print(f"坐标位置: 行 {row.item()} (变量: {v_obj.name}), 列 {col.item()}")
                print(f"特征详情: {graph.var_attributes[row]}")
                print(f"诊断: 如果此处出现 1e20，请检查 _get_feature_var 内部逻辑！")
                print("X"*60 + "\n")
        # ======================================================

            
        return var_global_to_local, cons_global_to_local

    #把 根 LP（原始问题）转成一个 图表示
    # def get_root_graph(self, model, device=None):
        
    #     dev = device if device != None else self.device
        
    #     graph = BipartiteGraphStatic0(self.n0, dev)
        
    #     self._add_vars_to_graph(graph, model, dev)
    #     self._add_conss_to_graph(graph, model, self.original_conss, dev)
    
        
    #     return graph
    
    
    # def _get_obj_adjacency(self, model):
    
    #    if self.obj_adjacency  == None:
    #        var_coeff = { self.var2idx[ str(t[0]) ]:c for (t,c) in model.getObjective().terms.items() if c != 0.0 }
    #        var_idxs = list(var_coeff.keys())
    #        weigths = list(var_coeff.values())
    #        cons_idxs = [0]*len(var_idxs)
           
    #        self.obj_adjacency =  torch.torch.sparse_coo_tensor([var_idxs, cons_idxs], weigths, (self.n0, 1), device=self.device)
    #        self.obj_adjacency = torch.hstack((-1*self.obj_adjacency, self.obj_adjacency))
           
    #    return self.obj_adjacency         
       
    
    # def _add_scip_obj_cons(self, model, sub_milp, graph):
    #     adjacency_matrix = self._get_obj_adjacency(model)
    #     cons_feature = torch.tensor([[ sub_milp.getEstimate() ], [ -sub_milp.getLowerbound() ]], device=self.device).float()
    #     graph.cons_block_idxs.append(len(self.all_conss_blocks_features))
    #     self.all_conss_blocks_features.append(cons_feature)
    #     self.all_conss_blocks.append(adjacency_matrix)
  
    
    # 把图中所有变量加进图中           
    # def _add_vars_to_graph(self, graph, model, device=None):
    #     #add vars
        
    #     dev = device if device != None else self.device
        
    #     for idx, var in enumerate(self.varrs):
    #         graph.var_attributes[idx] = self._get_feature_var(model, var, dev)


    # def _add_selected_vars_to_graph(self, graph, model, selected_vars, device=None):
    #     """
    #     只将 K-Hop 涉及到的变量添加到图中
    #     selected_vars: list of SCIP Variable objects
    #     """
    #     dev = device if device is not None else self.device
        
    #     # 注意：这里的 idx 必须和你在后续构建边（Edges）时的索引逻辑对应
    #     # 建议使用变量在原始模型中的 Index 或者建立一个映射
    #     for var in selected_vars:
    #         # 仅为选中的变量计算并存储特征
    #         v_idx = var.getIndex() # 或者你自定义的索引映射
    #         graph.var_attributes[v_idx] = self._get_feature_var(model, var, dev)


    def _add_conss_to_graph(self, graph, model, conss, device=None):
        
        dev = device if device != None else self.device

        if len(conss) == 0:
            return

        cons_attributes = torch.zeros(len(conss), graph.d1, device=dev).float()
        var_idxs = []
        cons_idxs = []
        weigths = []
        for cons_idx, cons in enumerate(conss):

            cons_attributes[cons_idx] =  self._get_feature_cons(model, cons, dev)
          
            for var, coeff in model.getValsLinear(cons).items():

                if str(var) in self.var2idx:
                    var_idx = self.var2idx[str(var)]
                elif 't_'+str(var) in self.var2idx:
                    var_idx = self.var2idx['t_' + str(var)]
                else:
                    var_idx = self.var2idx[ '_'.join(str(var).split('_')[1:]) ] 
                    
                var_idxs.append(var_idx)
                cons_idxs.append(cons_idx)
                weigths.append(coeff)


        adjacency_matrix =  torch.sparse_coo_tensor([var_idxs, cons_idxs], weigths, (self.n0, len(conss)), device=dev) 
        
        #add idx to graph
        graph.cons_block_idxs.append(len(self.all_conss_blocks_features)) #carreful with parralelization
        #add appropriate structure to self
        self.all_conss_blocks_features.append(cons_attributes)
        self.all_conss_blocks.append(adjacency_matrix)
      
    # sub_milp:节点
    # def _change_branched_bounds(self, graph, sub_milp):
        
    #     bvars, bbounds, btypes = sub_milp.getParentBranchings()
    #     # bvars, bbounds, btypes = getNodeVarBranchings(sub_milp)
        
    #     for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
            
    #         if str(bvar) in self.var2idx:
    #             var_idx = self.var2idx[str(bvar)]
    #         elif 't_'+str(bvar) in self.var2idx:
    #             var_idx = self.var2idx['t_' + str(bvar)]
    #         else:
    #             var_idx = self.var2idx[ '_'.join(str(bvar).split('_')[1:]) ] 
            
    #         graph.var_attributes[var_idx, int(btype) ] = bbound
            
    # def _change_branched_bounds(self, graph, sub_milp):
    #     # 1. 获取底层分支数据
    #     bvars, bbounds, btypes = sub_milp.getParentBranchings()
        
    #     for bvar, bbound, btype in zip(bvars, bbounds, btypes):
    #         name = bvar.name
            
    #         # 2. 极简匹配逻辑：按优先级尝试 原始名 -> 带 t_ 前缀 -> 去掉前缀后的名
    #         var_idx = self.var2idx.get(name) or \
    #                 self.var2idx.get(f"t_{name}") or \
    #                 (self.var2idx.get(name.split('_', 1)[-1]) if '_' in name else None)

    #         # 3. 更新特征 (btype: 0 为 LB, 1 为 UB)
    #         if var_idx is not None:
    #             graph.var_attributes[var_idx, int(btype)] = bbound

    def _change_branched_bounds_local(self, graph, sub_milp, var_map):
        # 1.  被分支的变量对象  分支设置的 bound 值  分支类型
        bvars, bbounds, btypes = sub_milp.getParentBranchings()
        SCIP_INF = 1e7
        
        for bvar, bbound, btype in zip(bvars, bbounds, btypes):
            name = bvar.name
            
            # 2. 极简匹配逻辑：按优先级尝试 原始名 -> 带 t_ 前缀 -> 去掉前缀后的名
            global_idx = self.var2idx.get(name) or \
                    self.var2idx.get(f"t_{name}") or \
                    (self.var2idx.get(name.split('_', 1)[-1]) if '_' in name else None)

            # 3. 更新特征 (btype: 0 为 LB, 1 为 UB)
            if global_idx is not None:
                local_idx = var_map.get(global_idx)
                if local_idx is not None:
                    val = bbound
                    is_inf = 0
                    if btype == 0: # SCIP_BOUNDTYPE_LOWER
                        is_inf = 1 if val <= -SCIP_INF else 0
                    else:          # SCIP_BOUNDTYPE_UPPER
                        is_inf = 1 if val >= SCIP_INF else 0

                    # 规则：如果是无穷大，数值位归 0
                    clean_val = val if not is_inf else 0.0

                    # 3. 写入特征矩阵 (严格对应 10 维索引)
                    # res[0]:sol, [1]:lb, [2]:ub, [3]:is_lb_inf, [4]:is_ub_inf ...
                    target_val_col = int(btype) + 1  # btype 0->col 1 (lb), 1->col 2 (ub)
                    target_flag_col = int(btype) + 3 # btype 0->col 3 (is_lb), 1->col 4 (is_ub)
                    # 写入数值
                    graph.var_attributes[local_idx, target_val_col] = clean_val
                    # 写入标志位
                    graph.var_attributes[local_idx, target_flag_col] = float(is_inf)

                    current_s = bvar.getLPSol()
                    is_s_inf = 1 if abs(current_s) >= SCIP_INF else 0
                    graph.var_attributes[local_idx, 0] = current_s if not is_s_inf else 0.0
                    graph.var_attributes[local_idx, 9] = float(is_s_inf)


                    
            
    #于给单个约束 cons 提取特征
    # def _get_feature_cons(self, model, cons, device=None):
        
    #     dev = device if device != None else self.device
        
    #     try:
            
    #         cons_n = str(cons) #把约束转换为字符串
    #         if re.match('flow', cons_n): #如果约束名字以 "flow" 开头
                
    #             rhs = model.getRhs(cons) #获取右端项 RHS
    #             leq = 0
    #             eq = 1  #表示该约束是等式约束
    #             geq = 0
    #         elif re.match('arc', cons_n): #如果约束名字以 "arc" 开头
    #             rhs = 0  #RHS 固定设为 0
    #             leq = eq =  1
    #             geq = 0
                
    #         else:
    #             rhs = model.getRhs(cons)
    #             leq = eq = 1
    #             geq = 0
    #     except:
    #         'logicor no repr'
    #         rhs = 0
    #         leq = eq = 1
    #         geq = 0
        
    #     # 在 return 前添加
    #     # if rhs > 1e15:
    #     #     print(f"!!! 捕捉到无穷大 RHS: 约束名={cons.name}, RHS={rhs}")
    #     #返回约束特征张量
    #     return torch.tensor([ rhs, leq, eq, geq ], device=dev).float()

    def _get_feature_cons(self, model, cons, device=None):
        dev = device if device is not None else self.device
        SCIP_INF = 1e19 

        
        lhs = model.getLhs(cons)
        rhs = model.getRhs(cons)

        # 初始化互斥标志位
        is_geq = 0   # x >= b (LHS有限, RHS无穷)
        is_leq = 0   # x <= b (LHS无穷, RHS有限)
        is_eq = 0    # x == b (LHS == RHS)
        is_range = 0 # a <= x <= b (LHS, RHS 均有限且不等)

        # 1. 逻辑判定 (互斥优先级)
        if abs(lhs - rhs) < 1e-9:
            is_eq = 1
        elif lhs > -SCIP_INF and rhs < SCIP_INF:
            is_range = 1
        elif lhs > -SCIP_INF:
            is_geq = 1
        elif rhs < SCIP_INF:
            is_leq = 1

        # 2. 数值脱敏 (防止 1e20 炸弹)
        # 将无穷大映射为 0，因为在对应的标志位下，无穷大没有数值意义
        clean_lhs = lhs if lhs > -SCIP_INF else 0.0
        clean_rhs = rhs if rhs < SCIP_INF else 0.0
        res = torch.tensor([clean_lhs, clean_rhs, is_geq, is_leq, is_eq, is_range], device=dev).float()
        
        # 修复：对数值型特征 (索引 0, 1) 进行对数压缩
        idx_to_log = [0, 1]
        res[idx_to_log] = torch.sign(res[idx_to_log]) * torch.log1p(torch.abs(res[idx_to_log]))
            
        # except Exception:
        #     # 默认兜底为等式 0
        #     clean_lhs, clean_rhs, is_geq, is_leq, is_eq, is_range = 0.0, 0.0, 0, 0, 1, 0

        # 返回 6 维特征
        return res

    #提取单个变量 var 的特征向量
    # def _get_feature_var(self, model, var, device=None):
        
    #     dev = device if device != None else self.device
    #     #获取变量的原始下界和上界
    #     lb, ub = var.getLbOriginal(), var.getUbOriginal()
        
    #     #裁剪，防止数据过大
    #     if lb <= - 0.999e+20:
    #         lb = -300
    #     if ub >= 0.999e+20:
    #         ub = 300
        
    #     #获取该变量在目标函数中的系数
    #     objective_coeff = model.getObjective()[var]
    #     #获取变量类型的 one-hot 编码，One-hot 编码将变量类型表示为一个长度为 3 的向量，其中只有对应其类型的位为 1，其余位为 0
    #     binary, integer, continuous = self._one_hot_type(var)
    
        
    #     return torch.tensor([ lb, ub, objective_coeff, binary, integer, continuous ], device=dev).float()

    def _get_feature_var(self, model, var, device=None):
        dev = device if device is not None else self.device
        
        # 1. 局部 Bound 替换原始 Bound (验证当前分枝状态)
        lb, ub = var.getLbLocal(), var.getUbLocal()
        
        # 2. 软截断 (将 1e20 映射到 10.0，减少方差冲击)
        SCIP_INF = 1e7
        is_lb_inf = 1 if lb <= -SCIP_INF else 0
        is_ub_inf = 1 if ub >= SCIP_INF else 0

        # 既然有了标志位，数值位就可以设为 0
        clean_lb = lb if not is_lb_inf else 0.0
        clean_ub = ub if not is_ub_inf else 0.0
        
        # 3. 性能优化：避免重复调用 getObjective()
        # 获取变量的当前目标值
        obj_coeff = var.getObj() 

        # 4. 类型编码 (原本的 one-hot 没问题)
        binary, integer, continuous = self._one_hot_type(var)
        
        # 5. 【新增验证点】LP 解
        sol = var.getLPSol()
        is_sol_inf = 1 if abs(sol) >= SCIP_INF else 0
        clean_sol = sol if not is_sol_inf else 0.0

        # 现在的特征维度变成 10 维
        res = torch.tensor([clean_sol, clean_lb, clean_ub, is_lb_inf, is_ub_inf,obj_coeff, binary, integer, continuous,is_sol_inf], device=dev).float()
        
        idx_to_log = [0, 1, 2, 5]
        res[idx_to_log] = torch.sign(res[idx_to_log]) * torch.log1p(torch.abs(res[idx_to_log]))
        # 6. 最后一道防线：数值脱敏
        return res
    
    
    def _one_hot_type(self, var):
        vtype = var.vtype()
        binary, integer, continuous = 0,0,0
        
        if vtype == 'BINARY':
            binary = 1
        elif vtype == 'INTEGER':
            integer = 1
        elif vtype == 'CONTINUOUS':
            continuous = 1
            
        return binary, integer,  continuous
        
        
#用来表示 二部图中“静态不变”的那一部分结构       
# class BipartiteGraphStatic0():
    
#     #Defines the structure of the problem solved. Invariant toward problems
#     #n0：变量数量
#     #d0=9：变量特征维度
#     #d1=6：约束特征维度
#     #allocate：是否立刻分配内存
#     def __init__(self, n0, device, d0=9, d1=6, allocate=True):
        
#         self.n0, self.d0, self.d1 = n0, d0, d1
#         self.device = device
        
#         if allocate:
#             self.var_attributes = torch.zeros(n0,d0, device=self.device)
#             self.cons_block_idxs = []
#         else:
#             self.var_attributes = None
#             self.cons_block_idxs = None
    
#     #定义一个 浅结构 + 深数据 的拷贝方法，用于搜索树展开
#     def copy(self):
        
#         #创建一个新的 BipartiteGraphStatic0 对象，不分配内存
#         copy = BipartiteGraphStatic0(self.n0, self.device, allocate=False)
        
#         copy.var_attributes = self.var_attributes.clone()
#         copy.cons_block_idxs = self.cons_block_idxs #no scip bonds
        
#         return copy

class BipartiteGraphStatic0():
    def __init__(self, n0, device, d0=10, d1=6, allocate=True):
        self.n0, self.d0, self.d1 = n0, d0, d1
        self.device = device
        
        # 记录约束数量（在 K-Hop 中，n1 是动态的）
        self.n_vars = n0
        # self.n_conss = 0 
        
        if allocate:
            self.var_attributes = torch.zeros(n0, d0, device=self.device)
            # 增加约束内存分配
            self.cons_attributes = None # 稍后根据提取到的约束数量动态分配
            self.cons_block_idxs = []
        else:
            self.var_attributes = None
            self.cons_attributes = None
            self.cons_block_idxs = None
            
        # 预留边存储空间
        self.local_edge_index = None
        self.local_edge_attr = None

    def copy(self):
        # 创建新对象
        new_copy = BipartiteGraphStatic0(self.n0, self.device, self.d0, self.d1, allocate=False)
        
        # 1. 复制变量特征
        if self.var_attributes is not None:
            new_copy.var_attributes = self.var_attributes.clone()
        
        # 2. 复制约束特征 (修复你之前的丢失问题)
        if hasattr(self, 'cons_attributes') and self.cons_attributes is not None:
            new_copy.cons_attributes = self.cons_attributes.clone()
            # new_copy.n_conss = self.n_conss
            
        # 3. 复制边信息 (K-Hop 模式下至关重要)
        if hasattr(self, 'local_edge_index') and self.local_edge_index is not None:
            new_copy.local_edge_index = self.local_edge_index.clone()
            new_copy.local_edge_attr = self.local_edge_attr.clone()
            
        new_copy.cons_block_idxs = self.cons_block_idxs
        
        return new_copy