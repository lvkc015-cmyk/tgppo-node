


import torch
import numpy as np
import time
import gc
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

        # 在初始化时，直接构建全局静态根图！   
        self.root_graph = self._build_global_root_graph(model)
        self.recorded[1] = self.root_graph

            
        
    # 【新增方法】：一次性构建全局二分图
    def _build_global_root_graph(self, model):
        dev = self.device
        #筛选线性约束
        linear_conss = [c for c in self.original_conss if c.isLinear()]
        n_vars = len(self.varrs)
        n_conss = len(linear_conss)
        
        graph = BipartiteGraphStatic0(n_vars, n_conss, dev, allocate=True)
        
        # 1. 建立“连续索引映射” (必须连续 0~N-1)
        var2local = {v.name: i for i, v in enumerate(self.varrs)}
        cons2local = {c.name: i for i, c in enumerate(linear_conss)}
        
        # 2. 填充全局变量初始特征
        for i, var in enumerate(self.varrs):
            graph.var_attributes[i] = self._get_feature_var(model, var, dev)
            
        # 3. 填充全局约束特征 & 构建全局边
        edge_indices = []
        edge_values = []
        for i, cons in enumerate(linear_conss):
            graph.cons_attributes[i] = self._get_feature_cons(model, cons, dev)
            
            # 连边
            for v_name, val in self.cons_to_coeffs.get(cons.name, {}).items():
                # 兼容你以前的命名逻辑
                clean_name = v_name[2:] if v_name.startswith("t_") else v_name
                if clean_name in var2local:
                    edge_indices.append([var2local[clean_name], i])
                    edge_values.append(val)
                    
        # 4. 转换并压缩边矩阵
        if edge_indices:
            graph.local_edge_index = torch.tensor(edge_indices, dtype=torch.long, device=dev).t().contiguous()
            raw_edge_attr = torch.tensor(edge_values, dtype=torch.float32, device=dev).unsqueeze(1)
            graph.local_edge_attr = torch.sign(raw_edge_attr) * torch.log1p(torch.abs(raw_edge_attr))
            
        return graph
    
    #极速获取结点的子图
    def record_sub_milp_graph(self, model, sub_milp, cands=None, k_hops=None):
        # 1. 如果已经记录过，直接返回
        node_num = sub_milp.getNumber()
        if node_num in self.recorded:
            return

        # 2. 找到父结点的图（递归溯源，直到根图）
        parent = sub_milp.getParent()
        if parent is None:
            # 万一它是根结点（理论上 __init__ 里已经建了，这是兜底）
            graph = self.root_graph.copy()
        else:
            parent_num = parent.getNumber()
            if parent_num not in self.recorded:
                self.record_sub_milp_graph(model, parent)  # 递归构建父图
            # 从父图极速拷贝（共享边和约束，只拷贝变量矩阵）
            graph = self.recorded[parent_num].copy()

        # 3. 极速状态注入：只更新发生了 Branching 的那个变量的 Bound！
        self._change_branched_bounds_global(graph, sub_milp)

        # 4. 存入内存库
        self.recorded[node_num] = graph


    # 把当前节点的 branching 决策（变量上下界变化）写进 graph 的变量特征里
    # sub_milp表示 当前 B&B 节点
    def _change_branched_bounds_global(self, graph, sub_milp):
        branchings = sub_milp.getParentBranchings()
        if branchings is None: 
            return
            
        bvars, bbounds, btypes = branchings ## 被分支的变量，分支值，类型（lower / upper）
        SCIP_INF = 1e7
        
        # 我们用 self.varrs 里的顺序作为全局矩阵的行索引
        var2local = {v.name: i for i, v in enumerate(self.varrs)}
        
        for bvar, bbound, btype in zip(bvars, bbounds, btypes):
            name = bvar.name
            clean_name = name[2:] if name.startswith("t_") else name
            
            global_idx = var2local.get(clean_name)
            
            if global_idx is not None:
                val = bbound
                #判断是否“无穷 bound”
                if btype == 0: # SCIP_BOUNDTYPE_LOWER
                    is_inf = 1 if val <= -SCIP_INF else 0
                else:          # SCIP_BOUNDTYPE_UPPER
                    is_inf = 1 if val >= SCIP_INF else 0

                clean_val = val if not is_inf else 0.0

                # 写入到 10 维特征矩阵中
                target_val_col = int(btype) + 1  # 1 为 lb, 2 为 ub
                target_flag_col = int(btype) + 3 # 3 为 is_lb_inf, 4 为 is_ub_inf
                
                # 更新特征并进行 log 压缩处理
                raw_tensor = torch.tensor([clean_val], device=self.device)
                log_val = torch.sign(raw_tensor) * torch.log1p(torch.abs(raw_tensor))
                
                graph.var_attributes[global_idx, target_val_col] = log_val[0]
                graph.var_attributes[global_idx, target_flag_col] = float(is_inf)


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
        

    # def record_sub_milp_graph(self, model, sub_milp, cands=None, k_hops=2):
       
    #     if sub_milp.getNumber() in self.recorded:
    #         return

    #     # 如果极端情况下 cands 为 None (例如 SCIP 初始状态)，我们主动获取一次种子
    #     if cands is None:
    #         print("候选变量为空")
    #         cands, _, _ = model.getCandsState(self.var_dim, self.branch_count)
    #         # 如果依然拿不到，截取前 100 个变量作为兜底种子（防止程序崩溃）
    #         if not cands:
    #             cands = model.getVars()[:100]
    #         else:
    #             cands = cands[:400] # 强制执行 400 种子截断

    #     # 此时产生的 graph 对象的维度是 [N_local]，索引是 [0...N_local-1]
    #     graph = self._extract_khop_manual(model, sub_milp, cands, k_hop=k_hops)

    #     self.recorded[sub_milp.getNumber()] = graph
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

        
        return graph

    # 把“选中的变量 / 约束”映射到一个紧凑的局部编号空间，并只为它们计算特征，填进图里。
    def _add_selected_entities_to_graph(self, graph, model, selected_vars, selected_conss, device):
       
        dev = device if device else self.device
        
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
            
       
        # 返回 6 维特征
        return res

   

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
        
        

class BipartiteGraphStatic0():
    def __init__(self, n_vars, n_conss, device, d0=10, d1=6, allocate=True):
        self.n_vars = n_vars
        self.n_conss = n_conss
        self.d0, self.d1 = d0, d1 #d0:变量维度，d1：约束维度
        self.device = device
        
        if allocate:
            self.var_attributes = torch.zeros(n_vars, d0, device=self.device)
            self.cons_attributes = torch.zeros(n_conss, d1, device=self.device)
            self.local_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.local_edge_attr = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        else:
            self.var_attributes = None
            self.cons_attributes = None
            self.local_edge_index = None
            self.local_edge_attr = None

    def copy(self):
        # 创建新对象，不分配内存
        new_copy = BipartiteGraphStatic0(self.n_vars, self.n_conss, self.device, self.d0, self.d1, allocate=False)
        
        # 1. 【核心】只深拷贝（clone）变量特征，因为只有变量的 Bound 会随着结点改变！
        if self.var_attributes is not None:
            new_copy.var_attributes = self.var_attributes.clone()
        
        # 2. 【核心】浅拷贝（共享内存）约束特征和边矩阵，节约 99% 的内存！
        new_copy.cons_attributes = self.cons_attributes
        new_copy.local_edge_index = self.local_edge_index
        new_copy.local_edge_attr = self.local_edge_attr
        
        return new_copy