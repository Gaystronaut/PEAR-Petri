
import torch
import numpy as np
import dgl
import scipy.sparse as sp
import math
import random
import copy
import matplotlib.pyplot as plt
import networkx as nx
import csv
import os


MIN_NUM = epsilon = 1e-10
def is_sparse(matrix, threshold=0.8):
    """ 判断矩阵是否稀疏 """
    zero_count = np.count_nonzero(matrix == 0)
    total_elements = matrix.size
    zero_ratio = zero_count / total_elements
    return zero_ratio > threshold

def shift_left(vec, start_pos):
    value = vec[start_pos]
    if start_pos < vec.size - 1:
        vec[start_pos:-1] = vec[start_pos + 1:]
    vec[-1] = 0
    return value

class State:
    def __init__(self, instance, cur_time, QT, QD, M, M_c, M_com, e, total_violated):
        """
        Build a State
        :param instance: the problem instance considered,
        :param M: the marking vector
        :param cur_time: the current time slot
        :param QT:
        :param QD:
        :param L:
        """
        self.instance = instance
        self.cur_time = cur_time
        self.QT = QT
        self.QD = QD
        # self.L = L
        self.M = M
        self.M_c = M_c
        self.M_com = M_com
        self.e = e
        #-----------------------------------
        # self.computation_energy_consumption = np.zeros(instance.N)  # 新增：记录每个设备的总能耗
        # self.communication_energy_consumption= np.zeros((instance.N))  # 新增：记录设备间通信能耗
        #-----------------------------------
        self.total_violated = total_violated
    def __repr__(self):
        return f"State: cur_time={self.cur_time}, QT={self.QT}, QD={self.QD}, M={self.M}, M_c={self.M_c}, M_com={self.M_com}, e={self.e}"

    def copy(self):
        # 创建一个新的 State 实例，其中包括所有属性的深层复制
        return State(
            instance=copy.deepcopy(self.instance),
            cur_time=self.cur_time,  # 假设 cur_time 是不可变类型，不需要深复制
            QT=copy.deepcopy(self.QT),
            QD=copy.deepcopy(self.QD),
            M=copy.deepcopy(self.M),
            M_c=copy.deepcopy(self.M_c),
            M_com=copy.deepcopy(self.M_com),
            e=copy.deepcopy(self.e),
            total_violated=copy.deepcopy(self.total_violated)
        )

class Environment:

    def __init__(self, instance, reward_scaling, max_time, args):
        """
        Initialize the PN environment
        :param instance: the problem instance considered,
        """

        self.B = np.random.uniform(25, 50) # bandwidth
        self.f = np.zeros(instance.h_max)
        self.kappa = np.zeros(instance.h_max)
        self.R = np.zeros([instance.N, instance.N])
        self.f[0] = np.random.uniform(3.0, 6.0)   #:low:1.5,high:3.0
        self.kappa[0] = np.random.uniform(1.0, 2.5)
        self.alpha = 10 #500 :10
        self.beta = -1

        if instance.h_max == 1:
            self.f[0] = np.random.uniform(5.0, 11.0)  #:low:1.0,high:2.5
            self.kappa[0] = np.random.uniform(2.0, 5.0)

        if instance.h_max == 2:
            self.f[1] = np.random.uniform(2.0, 5.0)  #:low:1.0,high:2.5
            self.kappa[1] = np.random.uniform(1.0, 2.5)

        #if instance.h_max == 3:

        self.P_c = np.zeros(instance.h_max)
        for h in range(instance.h_max):
            self.P_c[h] = self.kappa[h] * self.f[h] ** 3# 计算功耗遵循 立方频率-功耗关系 W/(Hz³)，所有设备的计算资源消耗是一样的

        # channel gain and the noise of wireless channel needed to be calculated here
        # self.R = self.B * math.log([1 + self.P_com, 2])
        self.P_com = np.random.uniform(0.1, 0.3)#通信功耗是设备无线传输数据的固定功率消耗，通常由硬件（如射频模块）决定。与速率无关
        channel_noise = np.random.normal(10, 2)

        for i in range(instance.N):
            for j in range(instance.N):
                self.R[i][j] = self.B * math.log(1 + self.P_com/channel_noise, 2)

        self.tao = 300
        self.e_max = 10000
        self.E = np.random.uniform(1000, 2000)
        self.task_probability = 1

        self.instance = instance
        self.reward_scaling = reward_scaling
        self.max_time = max_time
        self.vector_len = self.instance.K * self.instance.N * (self.instance.N + self.instance.h_max)
        self.L = np.zeros([self.instance.K, self.instance.N, self.instance.N + self.instance.h_max])
        # time for offloading and computing
        # L[k][i][j] denotes the time cost of task of type k in D_i to be expired,
        # j in [0,self.instance.N) denotes the p^o, otherwise p^e
        for k in range(self.instance.K):
            for i in range(self.instance.N):
                for j in range(self.instance.N):
                    #
                    self.L[k][i][j] = self.instance.s[k]/self.R[i][j]
                self.L[k][i][i] = 0
                for j in range(self.instance.N, self.instance.N+ self.instance.h_max):
                    self.L[k][i][j] = self.instance.c[k][j - self.instance.N]/self.f[j - self.instance.N]
        self.p_num = self.vector_len + 3 * self.instance.K * self.instance.N + 2 * self.instance.h_max * self.instance.N \
                 + self.instance.N ** 2
        self.t_num = self.instance.K * self.instance.N * 2 * (self.instance.N + self.instance.h_max) + len(self.instance.tc)
        self.n_node_feat = 2 * (1 + self.instance.k_bound)
        self.args = args

    def get_initial_environment(self):
        """
        Return the initial state of the DP formulation: we are at the city 0 at time 0
        :return: The initial state
        """

        #print(f"[INFO-Process: Now process get_initial_environment()")
        
        cur_time = 0
        # need initialization ?
        M = np.zeros([self.instance.K, self.instance.N, 3 + self.instance.N + self.instance.h_max], dtype=int)
        # computing and memory resources
        M_c = np.ones([self.instance.N, self.instance.h_max, 2], dtype=int)
        M_c[:, :, 1] = 2000
        # communication
        M_com = np.ones([self.instance.N, self.instance.N], dtype=int)
        e = np.zeros(self.instance.N)
        e += self.e_max

        # fill the start node with some tokens.

        # QT[k][i][j][c] denotes the time of the c-th token in task of type k in D_i to be available,
        # j in [0,self.instance.N) denotes the p^o, otherwise p^e
        QT = np.zeros([self.instance.K, self.instance.N, 3 + self.instance.N + self.instance.h_max, self.instance.k_bound], dtype=int)
        # QD[k][i][j][c] denotes the time of the c-th token in task of type k in D_i to be expired,
        # j in [0,self.instance.N) denotes the p^o, otherwise p^e
        QD = np.zeros([self.instance.K, self.instance.N, 3 + self.instance.N + self.instance.h_max, self.instance.k_bound], dtype=int)
        for source in self.instance.sources:
            for i in range(self.instance.N):
                M[source, i, self.instance.N + self.instance.h_max] = max(self.instance.k_bound - 2, 0)
                QD[source, i, self.instance.N + self.instance.h_max][0:max(self.instance.k_bound - 2, 0)] = self.tao + 1
        lambda_rate = 3
        time_length = 60

        # 生成每分钟到来的任务数
        tasks_per_minute = np.random.poisson(lambda_rate, time_length)

        return State(self.instance, cur_time, QT, QD, M, M_c, M_com, e, 0)

    def p_idx(self, k, i, j):
        return np.ravel_multi_index((k, i, j), (self.instance.K, self.instance.N, 3 + self.instance.N + self.instance.h_max))

    def t_idx(self, k, i, j):
        t_shape = (self.instance.K, self.instance.N, 2 * (self.instance.N + self.instance.h_max))
        return np.ravel_multi_index((k, i, j), t_shape)

    def make_nn_input(self, cur_state, mode):
        """
        Return a DGL graph serving as input of the neural network. Assign features on the nodes and on the edges
        :param cur_state: the current state of the DP model
        :param mode: on GPU or CPU
        :return: the DGL graph
        """
        #print(f"[make nn input starts [-------1------------]]")
        p_num = self.p_num
        t_num = self.t_num
        A = np.zeros([p_num, t_num])
        A_ = np.zeros([p_num, t_num])
        te_base = 2 * self.instance.N
        tc_base = self.instance.K * self.instance.N * 2 * (self.instance.N + self.instance.h_max)
        pc_base = self.vector_len + 3 * self.instance.K * self.instance.N
        pcom_base = pc_base + 2 * self.instance.h_max * self.instance.N
       
        for k in range(self.instance.K):
            for i in range(self.instance.N):
                # p^o
                for j in range(self.instance.N):
                    pi = self.p_idx(k, i, j)
                    A[pi][self.t_idx(k, i, j + self.instance.N)] = 1 # t^o2
                    A_[pi][self.t_idx(k, i, j)] = 1 # t^o1
                # p^e
                for j in range(self.instance.h_max):
                    pi = self.p_idx(k, i, j + self.instance.N)
                    A_[pi][self.t_idx(k, i, j + te_base)] = 1 # t^e1
                    A[pi][self.t_idx(k, i, j + self.instance.h_max + te_base)] = 1 # t^e2
                # p^S1
                j = self.instance.N + self.instance.h_max
                pi = self.p_idx(k, i, j)
                    # t_c
                for c in range(len(self.instance.tc)):
                    for succ_idx in self.instance.tc[c].succ:
                        if succ_idx == k:
                            A_[pi][c + tc_base] = 1
                    # t_o1
                for c in range(self.instance.N):
                    A[pi][self.t_idx(k, i, c)] = 1
                # p^S2
                j += 1
                pi = self.p_idx(k, i, j)
                    # t_o2
                for c in range(self.instance.N):
                    A_[pi][self.t_idx(k, i, c + self.instance.N)] = 1
                    # t_e1
                for c in range(self.instance.h_max):
                    A[pi][self.t_idx(k, i, c + te_base)] = 1
                # p^E
                j += 1
                pi = self.p_idx(k, i, j)
                    # t_e2
                for c in range(self.instance.h_max):
                    A_[pi][self.t_idx(k, i, c + te_base + self.instance.h_max)] = 1
                    # t_c
                for c in range(len(self.instance.tc)):
                    for pre_idx in self.instance.tc[c].pre:
                        if pre_idx == k:
                            A[pi][c + tc_base] = 1
                # p^c and p^m
               
                for h in range(self.instance.h_max):
                    # t^e1
                    i_pc = np.ravel_multi_index((i, h, 0), cur_state.M_c.shape)
                    i_pm = np.ravel_multi_index((i, h, 1), cur_state.M_c.shape)
                    A[pc_base + i_pc][self.t_idx(k, i, h + te_base)] = 1 # p^c
                    A[pc_base + i_pm][self.t_idx(k, i, h + te_base)] = 1 # p^m
                    # t^e2
                    A_[pc_base + i_pc][self.t_idx(k, i, h + te_base + self.instance.h_max)] = 1  # p^c
                    A_[pc_base + i_pm][self.t_idx(k, i, h + te_base + self.instance.h_max)] = 1  # p^m
                # p^com
                for c in range(self.instance.N):
                    i_pcom = np.ravel_multi_index((i, c), cur_state.M_com.shape)
                    # t^o1
                    A[pcom_base + i_pcom][self.t_idx(k, i, c)] = 1
                    # t^o2
                    A_[pcom_base + i_pcom][self.t_idx(k, i, c + self.instance.N)] = 1

        full_matrix = np.zeros((p_num + t_num, p_num + t_num))
        full_matrix[:p_num, p_num:p_num + t_num] = A
        full_matrix[p_num:p_num + t_num, :p_num] = A_.T
        if is_sparse(full_matrix):
            # 如果矩阵是稀疏的，使用 scipy 稀疏矩阵并转换为 DGL 图
            sparse_adj = sp.csr_matrix(full_matrix)
            g = dgl.from_scipy(sparse_adj)
        else:
            # 如果矩阵不是稀疏的，直接使用 numpy 数组创建 DGL 图
            src, dst = np.nonzero(full_matrix)
            g = dgl.graph((src, dst))
        # g = nx.from_numpy_matrix(full_matrix, create_using=nx.DiGraph())
        # assert(nx.is_directed_acyclic_graph(g))

        M_feature = cur_state.M.reshape(-1)
        # M_feature /= np.mean(M_feature)
        # mean_value = np.mean(M_feature)
        # M_feature = np.where(mean_value != 0, M_feature - mean_value, M_feature)

        M_feature = np.hstack((M_feature, cur_state.M_c.reshape(-1)))
        M_feature = np.hstack((M_feature, cur_state.M_com.reshape(-1)))

        L_feature = self.L.reshape(-1)
        # L_feature /= np.mean(L_feature)
        # mean_value = np.mean(L_feature)
        # L_feature = np.where(mean_value != 0, L_feature - mean_value, L_feature)

        L_feature = np.hstack((L_feature, np.zeros(3 * self.instance.K * self.instance.N)))
        L_feature = np.hstack((L_feature, np.zeros(2 * self.instance.N * self.instance.h_max + self.instance.N ** 2)))

        temp_len = self.vector_len + 3 * self.instance.K * self.instance.N
        QT_feature = cur_state.QT.reshape(temp_len, self.instance.k_bound)
        row_means = np.mean(QT_feature, axis=1)
        row_means = row_means[:, np.newaxis]
        # QT_feature /= row_means
        # print("row_means:",row_means.type)
        row_means = np.where(np.abs(row_means) < MIN_NUM, MIN_NUM, row_means)
        QT_feature = np.where(row_means != 0, (QT_feature + 0.0) / row_means, 0.0)


        QD_feature = cur_state.QD.reshape(temp_len, self.instance.k_bound)
        row_means = np.mean(QD_feature, axis=1)
        row_means = row_means[:, np.newaxis]
        # QT_feature /= row_means
        row_means = np.where(np.abs(row_means) < MIN_NUM, MIN_NUM, row_means)
        QD_feature = np.where(row_means != 0, QT_feature / row_means, 0.0)

        QT_feature = np.vstack((QT_feature, np.zeros((p_num - temp_len, self.instance.k_bound))))
        QD_feature = np.vstack((QD_feature, np.zeros((p_num - temp_len, self.instance.k_bound))))

        M_feature = np.hstack((M_feature, np.zeros(t_num)))
        L_feature = np.hstack((L_feature, np.zeros(t_num)))
        QT_feature = np.vstack((QT_feature, np.zeros((t_num, self.instance.k_bound))))
        QD_feature = np.vstack((QD_feature, np.zeros((t_num, self.instance.k_bound))))

        node_feature = M_feature
        node_feature = np.vstack((node_feature, L_feature))
        node_feature = np.vstack((node_feature, QT_feature.T))
        node_feature = np.vstack((node_feature, QD_feature.T))
        


        node_feat_tensor = torch.tensor(node_feature.T, dtype=torch.float32)
        
        # if mode == 'gpu' and torch.cuda.is_available():
        #     node_feat_tensor = node_feat_tensor.to('cuda')
        # if mode == 'gpu' and torch.cuda.is_available():
        #     node_feat_tensor = node_feat_tensor.to('cuda')
        #！!!!!!!!!!!!!!!!!!!!!强行注释，先把整个graph转移到GPU上面，因为10000就爆显存了
        tensor_device = node_feat_tensor.device
        g = g.to(tensor_device)
        # feeding features into the dgl_graph
        g.ndata['n_feat'] = node_feat_tensor
        g.edata['e_feat'] = torch.zeros(g.number_of_edges(), 2 * (1 + self.instance.k_bound)).to(tensor_device)

        return g

    def get_next_state_with_reward(self, cur_state, action):
        """
        Compute the next_state and the reward of the RL environment from cur_state when an action is done
        :return: the next state and the reward collected
        """
        cur_time = cur_state.cur_time
        QD = cur_state.QD
        QT = cur_state.QT
        M = cur_state.M
        M_c = cur_state.M_c
        M_com = cur_state.M_com
        e = cur_state.e
        total_violated  = cur_state.total_violated
        finished_task = 0
        #-----------------能量补充监控--------------------------        
        computation_energy_consumption = np.zeros(self.instance.N)  # 新增：记录每个设备的总能耗
        communication_energy_comsumption= np.zeros((self.instance.N))  # 新增：记录设备间通信能耗
        #-----------------能量补充监控--------------------------
               

        reward = 0.8 #-10 1 :0.8
        if action == 0:
            # reward = -5 #-50 -5 :-10
            cur_time += 1

            # update QD and QT
            for k in range(self.instance.K):
                for i in range(self.instance.N):
                    # task generation
                    if M[k][i][self.instance.N + self.instance.h_max] < self.instance.k_bound - 1 and \
                            self.instance.is_source(k):
                        new_M = min(np.random.poisson(self.args.lamda_rate, 1)[0] + M[k][i][self.instance.N + self.instance.h_max] \
                            , self.instance.k_bound)
                        # print(new_M)
                        # new_M = self.instance.k_bound
                        QD[k][i][self.instance.N + self.instance.h_max][M[k][i][self.instance.N + self.instance.h_max]:new_M]  = self.tao + 1
                        M[k][i][self.instance.N + self.instance.h_max] = new_M

                    # if self.instance.is_source(k):
                    #     QD[k][i][self.instance.N + self.instance.h_max][M[k][i][self.instance.N + self.instance.h_max]
                    #         :(self.instance.k_bound - 1)] = self.tao + 1
                    #     M[k][i][self.instance.N + self.instance.h_max] = (self.instance.k_bound - 1)
                    # p^o
                    for j in range(self.instance.N):
                        for c in range(M[k][i][j]):
                            QD[k][i][j][c] -= 1
                            if e[i] >= self.P_com:
                                communication_energy_comsumption[i]+= self.P_com#新增
                                e[i] -= self.P_com
                                QT[k][i][j][c] -= 1
                    # p^e
                    for j in range(self.instance.N, self.instance.N + self.instance.h_max):
                        for c in range(M[k][i][j]):
                            QD[k][i][j][c] -= 1
                            if e[i] >= self.P_c[j - self.instance.N]:
                                e[i] -= self.P_c[j - self.instance.N]
                                computation_energy_consumption[i] += self.P_c[j - self.instance.N]
                                QT[k][i][j][c] -= 1

                    #  expiration check
                    for j in range(self.instance.N + self.instance.h_max):
                        if np.any(QD[k][i][j][0:M[k][i][j]] <= 0):
                            zero_idx = np.where(QD[k][i][j][0:M[k][i][j]] <= 0)
                            zero_idx = zero_idx[0]
                            # eliminate the expired element
                            for c in zero_idx[::-1]:
                                M[k][i][j] -= 1
                                shift_left(QD[k][i][j], c)
                                shift_left(QT[k][i][j], c)
                                total_violated += 1

            # fire t^o_{z,2l,i,j} and t^e_{z,2l,i,j} as many as possible
            for k in range(self.instance.K):
                for i in range(self.instance.N):
                    # t^o_{z,2l,i,j}
                    for j in range(self.instance.N):
                        if np.any(QT[k][i][j][0:M[k][i][j]] <= 0):
                            zero_idx = np.where(QT[k][i][j][0:M[k][i][j]] <= 0)
                            zero_idx = zero_idx[0]
                            for c in zero_idx[::-1]:
                                if M[k][i][self.instance.N + self.instance.h_max + 1] >= self.instance.k_bound:
                                    break
                                # transfer the QD  and QT element of $p^o_{z,l,i,j}$ to $p^S_{z,2l,i}$
                                shift_left(QT[k][i][j], c)
                                QD[k][i][self.instance.N + self.instance.h_max + 1][M[k][i][self.instance.N + self.instance.h_max + 1]] = \
                                shift_left(QD[k][i][j], c)
                                M[k][i][self.instance.N + self.instance.h_max + 1] += 1
                                M[k][i][j] -= 1
                                if i != j :
                                    M_com[i][j] += 1
                    # t^e_{z,2l,i,h}
                    for j in range(self.instance.N, self.instance.N + self.instance.h_max):
                        if np.any(QT[k][i][j][0:M[k][i][j]] <= 0):
                            zero_idx = np.where(QT[k][i][j][0:M[k][i][j]] <= 0)
                            zero_idx = zero_idx[0]
                            for c in zero_idx[::-1]:
                                if M[k][i][self.instance.N + self.instance.h_max + 2] >= self.instance.k_bound:
                                    break
                                # transfer the QD and QT element of $p^e_{z,l,i,j}$ to $p^E_{z,l,i}$
                                shift_left(QT[k][i][j], c)
                                QD[k][i][self.instance.N + self.instance.h_max + 2][M[k][i][self.instance.N + self.instance.h_max + 2]] = \
                                shift_left(QD[k][i][j], c)
                                M[k][i][self.instance.N + self.instance.h_max + 2] += 1
                                M[k][i][j] -= 1
                                for h in range(self.instance.h_max):
                                    M_c[i][h][0] += 1
                                    M_c[i][h][1] += self.instance.s[k]
            # reward

            for k in self.instance.sinks:
                for i in range(self.instance.N):
                    # print("M(p^E): ", M[k][i][self.instance.N + self.instance.h_max + 2], "k", k, "i", i)
                    if M[k][i][self.instance.N + self.instance.h_max + 2] != 0:
                        # print("get reward!")
                        reward += self.alpha * M[k][i][self.instance.N + self.instance.h_max + 2]
                        finished_task += M[k][i][self.instance.N + self.instance.h_max + 2]
                        M[k][i][self.instance.N + self.instance.h_max + 2] = 0
            if self.args.count_violated:
                reward -= self.alpha * (total_violated - cur_state.total_violated)
            # fire t^c as many as possible
            for i in range(self.instance.N):
                for t in self.instance.tc:
                    is_firable = True
                    for k in t.pre:
                        if M[k][i][self.instance.N + self.instance.h_max + 2] < 1:
                            is_firable = False
                    if is_firable:
                        D_value = self.tao
                        for k in t.pre:
                            idx = np.argmin(QD[k][i][self.instance.N + self.instance.h_max + 2][0:M[k][i][self.instance.N + self.instance.h_max + 2]])
                            D_value = min(D_value, shift_left(QD[k][i][self.instance.N + self.instance.h_max + 2], idx))
                            M[k][i][self.instance.N + self.instance.h_max + 2] -= 1
                        for k in t.succ:
                            if M[k][i][self.instance.N + self.instance.h_max] >= self.instance.k_bound - 1:
                                continue
                            QD[k][i][self.instance.N + self.instance.h_max][M[k][i][self.instance.N + self.instance.h_max]] = D_value
                            M[k][i][self.instance.N + self.instance.h_max] += 1

            # battery recharge
            for i in range(self.instance.N):
                e[i] = min(e[i] + self.E, self.e_max)

            return State(self.instance, cur_time, QT, QD, M, M_c, M_com, e, total_violated), reward, finished_task,computation_energy_consumption,communication_energy_comsumption

        action -= self.p_num

        a = np.zeros([self.instance.K, self.instance.N, self.instance.N + self.instance.h_max])
        k, i, j = np.unravel_index(action, (self.instance.K, self.instance.N, 2 * (self.instance.N + self.instance.h_max)))
        if j < self.instance.N:
            a[k][i][j] = 1
        else:
            a[k][i][j - self.instance.N] = 1
        # action = a.reshape(self.instance.K, self.instance.N, self.instance.N + self.instance.h_max)
        idx = np.where(a == 1)
        k, i, j = list(zip(*idx))[0]

        # if the t chosen to be fired is t^o
        if j < self.instance.N:
            # print("action t^o  k:", k, "i:", i, "j:", j, "M:", M[k][i][j])
            idx = np.argmin(QD[k][i][self.instance.N + self.instance.h_max][0:M[k][i][self.instance.N + self.instance.h_max]])
            D_value = shift_left(QD[k][i][self.instance.N + self.instance.h_max], idx)
            M[k][i][self.instance.N + self.instance.h_max] -= 1
            # if M[k][i][j] >= self.instance.k_bound:
            #     return State(self.instance, cur_time, QT, QD, M, M_c, M_com, e), reward + self.beta
            # print("D_value", D_value)
            QD[k][i][j][M[k][i][j]] = D_value
            QT[k][i][j][M[k][i][j]] = self.L[k][i][j]
            M[k][i][j] += 1
            if i != j :
                M_com[i][j] -= 1
                communication_energy_comsumption[i] += self.P_com#新增
                e[i] -= self.P_com
                
            return State(self.instance, cur_time, QT, QD, M, M_c, M_com, e, total_violated), reward, finished_task,computation_energy_consumption,communication_energy_comsumption
        else: # if the t chosen to be fired is t^e
            # print("action t^e  k:", k, "i:", i, "j:", j, "M:", M[k][i][j])
            idx = np.argmin(QD[k][i][self.instance.N + self.instance.h_max + 1][0:M[k][i][self.instance.N + self.instance.h_max + 1]])
            D_value = shift_left(QD[k][i][self.instance.N + self.instance.h_max + 1], idx)
            M[k][i][self.instance.N + self.instance.h_max + 1] -= 1
            # if M[k][i][j] >= self.instance.k_bound:
            #     return State(self.instance, cur_time, QT, QD, M, M_c, M_com, e), reward + self.beta
            QD[k][i][j][M[k][i][j]] = D_value
            QT[k][i][j][M[k][i][j]] = self.L[k][i][j]
            M[k][i][j] += 1
            M_c[i][j - self.instance.N][0] -= 1
            M_c[i][j - self.instance.N][1] -= self.instance.s[k]
            e[i] -= self.P_c[j - self.instance.N]
            computation_energy_consumption[i] += self.P_c[j - self.instance.N]
            return State(self.instance, cur_time, QT, QD, M, M_c, M_com, e, total_violated), reward, finished_task,computation_energy_consumption,communication_energy_comsumption
        # if the energy cost of task completion exceeds the battery level,
        # return the negative reward and dessert the task (TBD)

    def get_valid_actions(self, cur_state):
        M = cur_state.M
        M_c = cur_state.M_c
        M_com = cur_state.M_com
        e = cur_state.e
        avail = np.zeros(self.p_num + self.t_num)
        available_idx = []

        for k in range(self.instance.K):
            for i in range(self.instance.N):
                for j in range(self.instance.N + self.instance.h_max):
                    if M[k][i][j] >= self.instance.k_bound - 1:
                        continue
                    # if the t chosen to be fired is t^o
                    # 修改思路：有to按照贪心返回to，无则返回te列表？
                    if j < self.instance.N:
                        # marking of p^S_{z,2l-1,i} and at least exist one token could be fired
                        if M[k][i][self.instance.N + self.instance.h_max] < 1:
                            # or not np.any(QT[k][i][j][0:M[k][i][self.instance.N + self.instance.h_max]] == 0):
                            continue
                        # marking of p^o_{z,l,i,j} is not full
                        if M[k][i][j] >= self.instance.k_bound:
                            # or not np.any(QT[k][i][j][0:M[k][i][self.instance.N + self.instance.h_max]] == 0):
                            continue
                        # marking of p^{com}_{i,j}
                        if i != j and M_com[i][j] < 1:
                            continue
                        # energy check
                        if i != j and e[i] < self.P_com:
                            continue
                        # print("get_action t^o  k:", k, "i:", i, "j:", j, "M:", M[k][i][j])
                        index = np.ravel_multi_index((k, i, j), (
                            self.instance.K, self.instance.N, 2 * (self.instance.N + self.instance.h_max)))
                        index += self.p_num
                        avail[index] = 1
                        available_idx.append(index)
                    else:  # if the t chosen to be fired is t^e
                        # marking of p^S_{z,2l,i}
                        if M[k][i][self.instance.N + self.instance.h_max + 1] < 1:
                            continue
                        # marking of p^e_{z,l,i,j} is not full
                        if M[k][i][j] >= self.instance.k_bound:
                            # or not np.any(QT[k][i][j][0:M[k][i][self.instance.N + self.instance.h_max]] == 0):
                            continue
                        # marking of p^c_{i,h} and energy check
                        if M_c[i][j - self.instance.N][0] < 1 or e[i] < self.P_c[j - self.instance.N]:
                            continue
                        # marking of p^m_{i,h}
                        if M_c[i][j - self.instance.N][1] < self.instance.s[k]:
                            continue
                        # print("get_action t^e  k:", k, "i:", i, "j:", j, "M:", M[k][i][j])
                        index = np.ravel_multi_index((k, i, j + self.instance.N), (
                        self.instance.K, self.instance.N, 2 * (self.instance.N + self.instance.h_max)))
                        index += self.p_num
                        avail[index] = 1
                        available_idx.append(index)

        return avail, np.array(available_idx)
    def plot_and_save(self, x, y, x_label, y_label, title, save_dir, file_name, show_plot=False):

        save_dir = f"{save_dir}/task_{self.args.n_task}_device_{self.args.n_device}_h_max_{self.args.n_h_max}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.figure()
        plt.plot(x, y, marker='o', linestyle='-')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)

        plt.legend([y_label])

        out_fig_file = os.path.join(save_dir, file_name)
        plt.savefig(out_fig_file)
        print(f"Plot saved to {out_fig_file}")

        if show_plot:
            plt.show()

        # 清理图形资源
        plt.clf()
        plt.close()

        out_csv_file = '%s/%s.csv' % (save_dir, title)

        with open(out_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([x_label, title])
            for i, j in zip(x, y):
                writer.writerow([i, j])



