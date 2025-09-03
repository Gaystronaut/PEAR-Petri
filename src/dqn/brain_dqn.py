
import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import numpy as np
import dgl

from src.graph_attention_network import GATNetwork


class BrainDQN:
    """
    Definition of the DQN Brain, computing the DQN loss
    """

    def __init__(self, args, num_node_feat, num_edge_feat):
        """
        Initialization of the DQN Brain
        :param args: argparse object taking hyperparameters
        :param num_node_feat: number of features on the nodes
        :param num_edge_feat: number of features on the edges
        环境状态 → 图构建 → GAT 嵌入 → 全连接层 → Q 值 → DQN 决策
        """

        self.args = args

        self.embedding = [(num_node_feat, num_edge_feat),
                         (self.args.latent_dim, self.args.latent_dim),
                         (self.args.latent_dim, self.args.latent_dim),
                         (self.args.latent_dim, self.args.latent_dim)]

        self.model = GATNetwork(self.embedding, self.args.hidden_layer, self.args.latent_dim, 1)
        self.target_model = GATNetwork(self.embedding, self.args.hidden_layer, self.args.latent_dim, 1)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        if args.load_bool:
            save_dir = f"{self.args.save_dir}/task_{self.args.n_task}_device_{self.args.n_device}_h_max_{self.args.n_h_max}"
            self.load(save_dir, args.saved_model_name)
            print("brain loaded!")

        if self.args.mode == 'gpu':
            self.model.cuda()
            self.target_model.cuda()

    def train(self, x, y):
        """
        Compute the loss between (f(x) and y)
        :param x: the input
        :param y: the true value of y
        :return: the loss
        """

        self.model.train()

        graph, _ = list(zip(*x))
        
        # 如果模式是 GPU，先将每个图移动到 GPU
        if self.args.mode == 'gpu' and torch.cuda.is_available():
            graph = [g.to('cuda') for g in graph]  # 每个图移动到 GPU
            self.model.to('cuda')  # 确保模型也在 GPU 上
        #print(f"[Brain-train]- graph.device: {graph[0].device}-self.model:{self.model.device}")

        graph_batch = dgl.batch(graph)

        y_pred = self.model(graph_batch, graph_pooling=False)
        y_pred = torch.stack([g.ndata["n_feat"] for g in dgl.unbatch(y_pred)]).squeeze(dim=2)
        y_tensor = torch.FloatTensor(np.array(y))

        if self.args.mode == 'gpu':
            y_tensor = y_tensor.contiguous().cuda()

        # loss = F.smooth_l1_loss(y_pred, y_tensor)
        # print("--- y_pred", y_pred, " ---")
        loss = F.l1_loss(y_pred, y_tensor,  reduction='sum')
        # print("--- learn_loss:", loss, " ---")

        # # 获取并打印 y_pred 的最小值和最大值
        # y_pred_min = y_pred.min().item()
        # y_pred_max = y_pred.max().item()
        #
        # # 获取并打印 y_tensor 的最小值和最大值
        # y_tensor_min = y_tensor.min().item()
        # y_tensor_max = y_tensor.max().item()
        # print("---MIN: Q_t:", y_tensor_min, " Q_e:", y_pred_min, "---")
        # print("---MAX: y_tensor:", y_tensor_max, "y_pred:", y_pred_max, "---")



        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, graph, target):
        """
        Predict the Q-values using the current graph, either using the model or the target model
        :param graph: the graph serving as input
        :param target: True is the target network must be used for the prediction
        :return: A list of the predictions for each node
        """
        if self.args.mode == 'gpu' and torch.cuda.is_available():
            graph = graph.to('cuda') #如果 graph 已经在 cuda 上，PyTorch/DGL 会检测到这一点。 不会执行任何实际的数据拷贝或移动操作。


        with torch.no_grad():
            if target:
                self.target_model.eval()
                res = self.target_model(graph, graph_pooling=False)
            else:
                self.model.eval()
                res = self.model(graph, graph_pooling=False)

        res = dgl.unbatch(res)
        return [r.ndata["n_feat"].data.cpu().numpy().flatten() for r in res]

    def update_target_model(self):
        """
        Synchronise the target network with the current one
        """

        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, folder, filename):
        """
        Save the model
        :param folder: Folder requested
        :param filename: file name requested
        """
        filepath = os.path.join(folder, filename)

        if not os.path.exists(folder):
            os.mkdir(folder)
            
        torch.save(self.model.state_dict(), filepath)

    def load(self, folder, filename):
        """
        Load the model
        :param folder: Folder requested
        :param filename: file name requested
        """

        filepath = os.path.join(folder, filename)
        # torch.load(self.model.state_dict(), filepath)
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(torch.load(filepath))
