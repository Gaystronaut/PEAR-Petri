# import matplotlib as mpl

# mpl.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import copy
import random
import time
import sys
import os
import csv
import pickle
import torch
from multiprocessing import Process, Queue, current_process,JoinableQueue
import queue
import numpy as np
import dgl
from src.Environment import Environment
from src.DQN.brain_dqn import BrainDQN
from src.DAG_PN import DAGGenerator
from src.util.prioritized_replay_memory import PrioritizedReplayMemory

#  definition of constants
MEMORY_CAPACITY = 700000#100000 #500000
GAMMA = 0.999
STEP_EPSILON = 5000.0
UPDATE_TARGET_FREQUENCY = 3 
VALIDATION_SET_SIZE = 100
RANDOM_TRIAL = 100
MAX_BETA = 10
MIN_VAL = -1000000
MAX_VAL = 1000000

reward_scaling=0.001

class TrainerDQN:
    """
    Definition of the Trainer DQN for the PN
    """

    def __init__(self, args):
        """
        Initialization of the trainer
        :param args:  argparse object taking hyperparameters and instance  configuration
        """

        self.args = args
        np.random.seed(self.args.seed)
        self.n_action = args.n_task * args.n_device * (args.n_device + args.n_h_max)

        self.num_node_feats = 2 * (1 + args.k_bound)
        self.num_edge_feats = 2 * (1 + args.k_bound)
        self.reward_scaling = 0.001

        self.brain = BrainDQN(self.args, self.num_node_feats, self.num_edge_feats)
        self.memory = PrioritizedReplayMemory(MEMORY_CAPACITY)
        
        print("[INFO] n_node_feat: %d" % MEMORY_CAPACITY)

        self.steps_done = 0
        self.init_memory_counter = 0
        self.n_step = args.n_step
        self.n_time_slot = args.n_time_slot  # number of time slots to end
        self.validation_len = 1

        print("***********************************************************")
        print("[INFO] NUMBER OF FEATURES")
        print("[INFO] n_node_feat: %d" % self.num_node_feats)
        print("[INFO] n_edge_feat: %d" % self.num_edge_feats)
        print("***********************************************************")

    def run_training(self):
        """
        Run de main loop for training the model
        """
        #  Generate a random instance
        instance = DAGGenerator(self.args.n_task, self.args.n_edge, self.args.n_h_max, self.args.k_bound, self.args.n_device, self.args.seed)
        print("[INFO] DAGGenerator is ready in trainer_dqn: ",instance)
        env = Environment(instance, self.reward_scaling, self.n_step, self.args)
        print("[INFO] Environment is ready in trainer_dqn")
        
        start_time = time.time()
        if self.args.plot_training:
            iter_list = []
            reward_list = []
            finished_job_list = []
            training_iter = []
            training_reward = []
            training_finished_job = []
            training_loss = []
            #--------新增----------
            avg_computing_energy =[]
            avg_communication_energy =[]
            avg_all_energy=[]
            training_time=[]
            avg_computing_energy_multiprocess=[]
            avg_communication_energy_multiprocess=[]
            avg_all_energy_multiprocess=[]
            training_time_multiprocess=[]

       
        if not self.args.use_random:
            if self.args.load_memory_bool==False:
                self.initialize_memory(env)#随机经验初始化, 这个时候还会从n_episode=0开始
                print(f"************************Now DQN's random initialize_memory is end*************************************************")
            else:
                self.load_initialized_memory()
                                             

        cur_best_reward = MIN_VAL

        if self.args.use_random:
            for i in range(self.args.n_episode):
                print(f"************************Now use_random n_episode is {i},This is round {i % 10} of random choice********************************")
                eps_start_time = time.time()
                total_eps_loss, beta, eps_reward_array, eps_job_array, eps_computing_energy_array, eps_communication_energy_array = self.run_episode(i, False, env)

                eps_job = np.mean(eps_job_array)
                eps_reward = np.mean(eps_reward_array)
                eps_computing_energy = np.mean(eps_computing_energy_array)
                eps_communication_energy = np.mean(eps_communication_energy_array)

                eps_end_time = time.time()
                eps_time=eps_end_time-eps_start_time

                avg_computing_energy.append(eps_computing_energy)
                avg_communication_energy.append(eps_communication_energy)
                avg_all_energy.append(eps_computing_energy+eps_communication_energy)
                training_time.append(eps_time)

                training_iter.append(i)
                training_reward.append(eps_reward)
                training_finished_job.append(eps_job)
                training_loss.append(total_eps_loss)   

                if self.args.plot_training and i %10==0:
                    print(f"[INFO] random_n_episode[{i}] is plotted.")
                                                

                    self.plot_and_save(training_iter, training_reward, 'Iteration', \
                                       'random_reward', 'random_reward', \
                                       self.args.save_dir, f'random_reward_greedy_{self.args.use_greedy}.png')
                    self.plot_and_save(training_iter, training_finished_job, 'Iteration', \
                                       'random_finished_job', 'random_finished_job', \
                                       self.args.save_dir, f'random_finished_job_greedy_{self.args.use_greedy}.png')
                    
                    

                    self.plot_and_save(training_iter, avg_computing_energy, 'Iteration', \
                                       'random_avg_computing_energy', 'random_avg_computing_energy', \
                                       self.args.save_dir, f'random_avg_computing_energy_greedy_{self.args.use_greedy}.png')
                    self.plot_and_save(training_iter, avg_communication_energy, 'Iteration', \
                                       'random_avg_communication_energy', 'random_avg_communication_energy', \
                                       self.args.save_dir, f'random_avg_communication_energy_greedy_{self.args.use_greedy}.png')
                    self.plot_and_save(training_iter, avg_all_energy, 'Iteration', \
                                       'random_avg_all_energy', 'random_avg_all_energy', \
                                       self.args.save_dir, f'random_avg_all_energy_greedy_{self.args.use_greedy}.png')
                    self.plot_and_save(training_iter, training_time, 'Iteration', \
                                       'random_training_time', 'random_training_time', \
                                       self.args.save_dir, f'random_training_time_greedy_{self.args.use_greedy}.png')
                
            return
        
    
        for i in range(self.args.n_episode):#此时采用的是DQN
            print(f"************************Now use [DQN_episode](n_episode={self.args.n_episode}) is {i},the exiting memory total is {self.memory.size()}, count memory is {self.memory.sumtree.count()}_This is round {i%10} of training**********************************")
            eps_start_time = time.time()
            #loss, beta, eps_reward, eps_job,eps_computing_energy,eps_communication_energy = self.run_episode(i, False, env)
            # 调用 run_episode，接收返回的数组
            total_eps_loss, beta, eps_reward_array, eps_job_array, eps_computing_energy_array, eps_communication_energy_array = self.run_episode(i, False, env)

            eps_job = np.mean(eps_job_array)
            eps_reward = np.mean(eps_reward_array)
            eps_computing_energy = np.mean(eps_computing_energy_array)
            eps_communication_energy = np.mean(eps_communication_energy_array)
                      

            eps_end_time = time.time()
            eps_time=eps_end_time-eps_start_time

            #-----------------------------
            training_iter.append(i)
            training_reward.append(eps_reward)
            training_finished_job.append(eps_job)
            training_loss.append(total_eps_loss)

            avg_computing_energy.append(eps_computing_energy)
            avg_communication_energy.append(eps_communication_energy)
            avg_all_energy.append(eps_computing_energy+eps_communication_energy)

            training_time.append(eps_time)

            avg_computing_energy_multiprocess.append(eps_reward_array)
            avg_communication_energy_multiprocess.append(eps_job_array)
            avg_all_energy_multiprocess.append(eps_computing_energy_array)
            training_time_multiprocess.append(eps_communication_energy_array)  
            #-------------------------
            
            if self.args.plot_training and i %10==0:
                print(f"[INFO] DQN_n_episode[{i}] is plotted.")                               

                self.plot_and_save(training_iter, training_reward, 'Iteration', \
                              'training_reward', 'training_reward', \
                              self.args.save_dir, 'training_reward.png')
                self.plot_and_save(training_iter, training_finished_job, 'Iteration', \
                              'training_finished_job', 'training_finished_job', \
                              self.args.save_dir, 'training_finished_job.png')
                self.plot_and_save(training_iter, training_loss, 'Iteration', \
                              'training_loss', 'training_loss', \
                              self.args.save_dir, 'training_loss.png')
                
                

                self.plot_and_save(training_iter, avg_computing_energy, 'Iteration', \
                              'avg_computing_energy', 'avg_computing_energy', \
                              self.args.save_dir, 'avg_computing_energy.png')
                self.plot_and_save(training_iter, avg_communication_energy, 'Iteration', \
                              'avg_communication_energy', 'avg_communication_energy', \
                              self.args.save_dir, 'avg_communication_energy.png')
                self.plot_and_save(training_iter, avg_all_energy, 'Iteration', \
                              'avg_all_energy', 'avg_all_energy', \
                              self.args.save_dir, 'avg_all_energy.png')
                                        

                self.only_save(training_iter, avg_computing_energy_multiprocess, 'Iteration','eps_reward_array', 'eps_reward_array', self.args.save_dir, 'eps_reward_array.png')
                self.only_save(training_iter, avg_communication_energy_multiprocess, 'Iteration','eps_job_array', 'eps_job_array', self.args.save_dir, 'eps_job_array.png')
                self.only_save(training_iter, avg_all_energy_multiprocess, 'Iteration','eps_computing_energy_array', 'eps_computing_energy_array', self.args.save_dir, 'eps_reward_array.png')
                self.only_save(training_iter, training_time_multiprocess, 'Iteration','eps_communication_energy_array', 'eps_communication_energy_array', self.args.save_dir, 'eps_reward_array.png')

            print(f"[INFO] DQN_n_episode[{i}] is plotted and use {eps_time}s.")

            #  We first evaluate the validation step every 10 episodes, until 100, then every 100 episodes.
            if (i % 25 == 0 and i < 101) or i % 25 == 0:
                print(f"[INFO] DQN_n_episode[{i}] begin  evaluate_instance ")
                start_evaluate_time=time.time()
                avg_finished_job = 0.0
                avg_reward = 0.0
                for j in range(self.validation_len):
                    tmp_r, tmp_t = self.evaluate_instance(j, env)
                    avg_reward += tmp_r
                    avg_finished_job += tmp_t

                avg_reward /= self.validation_len
                avg_finished_job /= self.validation_len

                cur_time = round(time.time() - start_time, 2)

                print('[DATA]', i, cur_time, avg_reward, total_eps_loss, beta)
                end_evaluate_time=time.time()
                evaluate_time=end_evaluate_time-start_evaluate_time
                print(f"[INFO] DQN_n_episode[{i}] evaluate_instance is end with {evaluate_time}s ")

                #sys.stdout.flush()

                if self.args.plot_training:
                    print(f"[INFO] DQN_n_episode[{i}] evaluating is plotted.")
                    iter_list.append(i * self.n_step)
                    reward_list.append(avg_reward)
                    finished_job_list.append(avg_finished_job)
                    self.plot_and_save(iter_list, reward_list, 'Iteration', \
                                  'evaluating_reward', 'evaluating_reward', \
                                  self.args.save_dir, 'evaluating_reward.png')
                    self.plot_and_save(iter_list, finished_job_list, 'Iteration', \
                                  'evaluating_finished_job', 'evaluating_finished_job', \
                                  self.args.save_dir, 'evaluating_finished_job.png')

                fn = "iter_%d_model.pth.tar" % i

                #  We record only the model that is better on the validation set wrt. the previous model
                #  We nevertheless record a model every 10000 episodes
                save_dir = f"{self.args.save_dir}/task_{self.args.n_task}_device_{self.args.n_device}_h_max_{self.args.n_h_max}_multi_{self.args.num_of_process}"
                if avg_reward >= cur_best_reward:
                    cur_best_reward = avg_reward
                    self.brain.save(folder=save_dir, filename=fn)
                elif i % 50 == 0:
                    self.brain.save(folder=save_dir, filename=fn)
          

    def initialize_memory(self, env):
        """
        Initialize the replay memory with random episodes and a random selection
        """
        int_temp=0

        # print("------------------start initialize_memory into ={MEMORY_CAPACITY},will use Random first-----------------")
        # while self.init_memory_counter < MEMORY_CAPACITY:
        #     int_temp+=1            
        #     self.run_episode(0, True, env)
        #     print(f"This the [{int_temp}] initialize_memory : continue run_episode(0, True, env) until init_memory_counter={self.init_memory_counter} reach MEMORY_CAPACITY={MEMORY_CAPACITY}")        
        print("------------------start initialize_memory ,will use Random first,only run once!!!-----------------")          
        self.run_episode(0, True, env)
        int_temp=int_temp+1
        print(f"This the [{int_temp}] initialize_memory : Now  initialize_memory size is {self.memory.size()} init_memory_counter={self.init_memory_counter} reach MEMORY_CAPACITY={MEMORY_CAPACITY}")
        init_loss = self.learning()  # learning procedure
        
        print("[INFO] Memory Initialized Loss:", init_loss)
        print("[INFO] Memory Initialized")

        # 保存初始化memory的代码
        if not os.path.exists(self.args.save_dir_initiliaze_memory_data):
            os.makedirs(self.args.save_dir_initiliaze_memory_data)
        
        filename = f"task_{self.args.n_task}_device_{self.args.n_device}_h_max_{self.args.n_h_max}_MEMORY_CAPACITY_{MEMORY_CAPACITY}.pkl"
        save_path = os.path.join(self.args.save_dir_initiliaze_memory_data, filename)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.memory, f)
        
        print(f"[INFO]Initialized memory saved to {save_path}")
        print("--------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------")

    # 新增的load_initialized_memory函数
    def load_initialized_memory(self):
        """Load pre-initialized memory from file"""
        # 构建文件路径
        filename = f"task_{self.args.n_task}_device_{self.args.n_device}_h_max_{self.args.n_h_max}_MEMORY_CAPACITY_{MEMORY_CAPACITY}.pkl"
        load_path = os.path.join(self.args.save_dir_initiliaze_memory_data, filename)
        
        # 检查文件是否存在
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Initialized memory file not found: {load_path}")
        
        # 加载memory
        with open(load_path, 'rb') as f:
            self.memory = pickle.load(f)
        
        # 更新memory计数器
        self.init_memory_counter = len(self.memory)
        
        print(f"Loaded initialized memory from {load_path}")
        print(f"Memory size: {len(self.memory)}")
        
        # 进行初始学习
        init_loss = self.learning()
        print("[INFO] Loaded Memory Initialized Loss:", init_loss)
        
        return init_loss
    
    # def run_episode(self, episode_idx, memory_initialization, env, is_greedy=False):
    #     if memory_initialization==True:
    #         self.run_init_episode(self, episode_idx, memory_initialization,env, is_greedy=False)
    #     if memory_initialization==False:
    #         self.run_train_episode(self, episode_idx, memory_initialization,env, is_greedy=False)

    


    def run_episode(self, episode_idx, memory_initialization, env,is_greedy=False):
        """
        修改后的 run_episode 函数，支持多进程并行收集经验
        return： total_episode_loss, temperature, total_reward, total_finished_job, mean_computation_energy, mean_communication_energy（都是self.args.num_of_process数量的矩阵）
        """
        # 初始化队列和进程池
        

        
        processes = []
        initial_env = copy.deepcopy(env)  # 深拷贝环境初始状态
        temperature = max(0., min(self.args.max_softmax_beta,
                                  (episode_idx - 1) / STEP_EPSILON * self.args.max_softmax_beta))
        UNDERSAMPLE_RATE= random.uniform(self.args.undersample_lower_bound, self.args.undersample_upper_bound)
        num_of_process=self.args.num_of_process

        total_learing_time=0.0
        total_add_time=0.0
        total_episode_loss=0.0


        # 启动 4 个进程，每个进程使用不同的随机种子,其他的参数如下采样率，要尽可能一样，保证数据同分布
        if memory_initialization==True:
            num_of_process=num_of_process*1 #random  随机初始化的时候 效率要乘以4倍,经过验证，24条进程就会爆显存，因为基本上一个进程就是1GB，刚刚好18个进程是最稳的。
            self.n_step=1000
        #queue_pakage = Queue()
        queue_pakage=JoinableQueue()
        for i in range(num_of_process):#默认是4
            seed = self.args.seed + i + 1  # 确保每个进程的随机种子不同
            process_id=i
            p = Process(
                target=TrainerDQN.worker_run_env,
                args=(
                    queue_pakage,
                    seed,
                    process_id,
                    initial_env,
                    self.brain,
                    #self.instance,
                    temperature,
                    UNDERSAMPLE_RATE,
                    self.args,
                    episode_idx,
                    memory_initialization,
                    is_greedy,
                )
            )
            p.daemon=True
            p.start()
            processes.append(p)
            print(f"[INFO] the process_id = {i} starts run")

        print(f"[INFO] All {num_of_process} process starts run")

        # 主线程维护 all_praocesses_new_sample 数组
        final_stats = []
        multi_id_loss=np.zeros(num_of_process)
        num_temp_nowait=0
        time_waiting=0
        num_sample=0
        num_sample_pakage=0

        while any(p.is_alive() for p in processes):
            #print(f"[INFO] processes[1].is_alive() is: {processes[1].is_alive()} ") 
            all_processes_dead = not any(p.is_alive() for p in processes)
            # 如果所有进程结束且队列为空，退出循环
            if all_processes_dead and queue_pakage.empty():
                print("[INFO] 所有子进程已结束且队列已空，退出循环")
                break          
            try:                
                                
                data = queue_pakage.get(block=True, timeout=1)
                num_temp_nowait +=1
                #print(f"[INFO] num_temp_nowait : {num_temp_nowait} ")

                #非阻塞获取队列——它会从队列中取出最早放入的一个元素（FIFO）
                #数据接收部分
                #应该每次处理process_batch_size=64的话，然后训练1次还可以。然后经验池要尽可能大，大约10倍吧，然后就不用区分不同的process id了
                if data["type"] == "data":
                    process_id = data["process_id"]
                    sample_package = data["sample_package"]
                    queue_pakage.task_done()  
                    num_sample_pakage=num_sample_pakage+1  
                    print(f"[INFO] Receive num_sample_pakage - [ {num_sample_pakage} -from Process ID : {process_id} ] " )            
                    #sample_package.append(sample)
                    for sample in sample_package:
                        
                        num_sample=num_sample+1
                        add_time_start=time.perf_counter()                        
                        if memory_initialization==True:
                            
                            error = abs(sample[2])  # reward在元组中是第3个元素（索引2）
                            if error==0:
                                print(f"[ERROR-Process ID : {process_id} - {os.getpid()}] sample[2]  : {sample[2]},error : {error}")                                
                            #error = abs(sample["reward"])  # the error of the replay memory is equals to the reward, at initialization
                            
                            self.init_memory_counter += 1                                                     
                            self.memory.add(error, sample)                                
                            multi_id_loss[process_id] += error
                            if self.init_memory_counter % 1000==0:
                                print(f"[INFO] The added sample: {self.init_memory_counter},the memory size is {self.memory.size()}, total_add_time :{total_add_time}")
                            #if self.init_memory_counter == MEMORY_CAPACITY:
                            #step_loss = 0
                        else:                               
                            x, y, errors = self.get_targets([(0, sample, 0)])  # feed the memory with the new samples
                            error = errors[0]
                            multi_id_loss[process_id] += error
                            #print(f"[INFO] add-Process ID : {process_id}:self.memory.is_full is {self.memory.is_full})")

                            if self.memory.is_full==True:
                                self.memory.random_add(error, sample) #sample已经是随机的了
                            else:
                                self.memory.add(error, sample)
                                #print(f"[INFO] add-Process ID : {process_id} - {os.getpid()}]:num_sample is {num_sample} ")

                            if num_sample % 1000==0:
                                print(f"[INFO - add] The num_sample: {num_sample},The memory size is {self.memory.size()}, atotal_add_time :{total_add_time},num_sample_pakage is {num_sample_pakage}")
                            #step_loss = self.learning()  # learning procedure
                        add_time_end=time.perf_counter()
                        add_time=add_time_end-add_time_start
                        total_add_time +=add_time

                        if num_sample % self.args.learing_frequence== 0:#考虑到GPU算的比较，快而且显存占用基本已经可以忽略不计算了，那直接把batchsize设置为了256。为了更新的频次，继续选择64
                        #if num_sample % self.args.batch_size == 0: #每收集到batchsiez数量的经验，就进行一次计算。大约时间是原先70分钟的32分之一然后乘以6个线程的数量，最后是12分钟左右                           
                            learing_time_start=time.perf_counter()
                            #虽然每次来回转移CPU和GPU需要大量时间，但是总体两级是两分钟，估计来回转移花不了太多时间。而且这个GPU的计算过程是远远快过CPU探索的速度的，感觉可以直接试一下转移到CPU上面去然
                            if memory_initialization:                        
                                step_loss = 0
                            else:
                                step_loss = self.learning()  # learning procedure
                                #print(f"[INFO] [step_loss = self.learning()]-Process ID : {process_id} - {os.getpid()}]:num_sample is {num_sample}，num_sample_pakage is {num_sample_pakage} ")
                            learing_time_end=time.perf_counter()
                            learing_time=learing_time_end-learing_time_start
                            total_learing_time +=learing_time                        
                            total_episode_loss += step_loss
                        

                    print(f"[INFO] DONE pakage-[ {num_sample_pakage}-from ID : {process_id}]sample-{num_sample} done,The memory size: {self.memory.sumtree.count()}, total_add_time :{round(total_add_time,4)},learning_time(every {self.args.learing_frequence}):{round(total_learing_time,4)}, undone pakage {queue_pakage.qsize()}")
                    #print(f"[INFO] Finish learing-{total_learing_time}s and adding-[{total_add_time}]s and step_loss")                    

                elif data["type"] == "end":
                    # queue.put({
                    #     "type": "end",
                    #     "process_id":process_id,
                    #     "temperature":temperature,
                    #     "total_reward":total_reward,               
                    #     "total_finished_job": total_finished_job,
                    #     "total_computation_energy": total_computation_energy,
                    #     "total_communication_energy": total_communication_energy
                    #     })
                    process_id = data["process_id"]
                    #final_stats[process_id] = data  # 存储该 process 的统计信息,结构类型是整个接收报文 
                    final_stats.append(data)
                    queue_pakage.task_done()

            except queue.Empty:
                if all_processes_dead:
                    print("[INFO] 所有子进程已结束，退出循环")
                    break
                else:
                    time_waiting+=1
                    if time_waiting % 120 ==0:
                        print(f"[INFO]已经等待了{time_waiting}秒，数据传输太慢")
                    continue  # 仍有存活的子进程，继续等待
            except Exception as e:                
                raise  # 队列为空时跳过
            except ConnectionResetError:  # 子进程崩溃导致队列连接断开
                print("[ERROR] 子进程崩溃，队列连接被重置！")
                # 检查哪些子进程还存活
                alive_processes = [p.is_alive() for p in processes]
                print(f"[DEBUG] 存活的子进程: {alive_processes}")
                # 终止所有子进程并退出
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                raise RuntimeError("子进程崩溃，训练终止")

        
        print(f"[INFO]************All processs is dead.Finish learing-{total_learing_time}s and adding-[{total_add_time}]s and step_loss")

        for p in processes:
            p.join()

        # 合并统计信息（例如总奖励、完成任务数），平均每个场景下的总奖励
        # #queue.put({
        #         "type": "end",
        #         "process_id":process_id,
        #         "temperature":temperature,
        #         "total_reward":total_reward,               
        #         "total_finished_job": total_finished_job,
        #         "total_computation_energy": total_computation_energy,
        #         "total_communication_energy": total_communication_energy
        #         })        
        # 直接返回每个子进程process id的合并统计信息
        total_reward_array = np.array([s["total_reward"] for s in final_stats])
        total_finished_job_array = np.array([s["total_finished_job"] for s in final_stats])
        mean_computation_energy_array = np.array([s["total_computation_energy"] for s in final_stats])
        mean_communication_energy_array = np.array([s["total_communication_energy"] for s in final_stats])

        total_reward=total_reward_array
        total_finished_job=total_finished_job_array
        mean_computation_energy=mean_computation_energy_array
        mean_communication_energy=mean_communication_energy_array

        
        if memory_initialization:
            print("this episode memory_initialization_memory_count:", self.memory.sumtree.count())
        print("result-total_loss:", total_episode_loss)

        return total_episode_loss, temperature, total_reward, total_finished_job, mean_computation_energy, mean_communication_energy

    # # ======================================并行化修改====================================================


    def evaluate_instance(self, idx, env):
        """
        Evaluate an instance with the current model
        :param idx: the index of the instance in the validation set
        :return: the reward collected for this instance
        """

        # instance = self.validation_set[idx]
        cur_state = env.get_initial_environment()

        total_reward = 0
        total_task = 0
        while True:
            graph = env.make_nn_input(cur_state, self.args.mode)
            # print("interval ————————————")
            avail, avail_idx = env.get_valid_actions(cur_state)
            if avail_idx.size != 0:
                action = self.select_action(graph, avail)
            else:
                action = 0
            cur_state, reward, finished_task,*_ = env.get_next_state_with_reward(cur_state, action)
            # print(cur_state)
            total_reward += reward
            total_task += finished_task
            if cur_state.cur_time >= self.n_time_slot :
                break

        return total_reward, total_task

    def select_action(self, graph, available):
        """
        Select an action according the to the current model
        :param graph: the graph (first part of the state)
        :param available: the vector of available (second part of the state)
        :return: the action, following the greedy policy with the model prediction
        """

        batched_graph = dgl.batch([graph, ])
        available = available.astype(bool)
        out = self.brain.predict(batched_graph, target=False)[0].reshape(-1)

        action_idx = np.argmax(out[available])

        action = np.arange(len(out))[available][action_idx]

        return action

    def soft_select_action(self, graph, available, beta):
        """
        Select an action according the to the current model with a softmax selection of temperature beta
        :param available: the vector of available
        :param beta: the current temperature
        :return: the action, following the softmax selection with the model prediction
        """
    
        batched_graph = dgl.batch([graph, ])
        available = available.astype(bool)
        out = self.brain.predict(batched_graph, target=False)[0].reshape(-1)
    
        if len(out[available]) > 1:
            logits = (out[available] - out[available].mean())
            div = ((logits ** 2).sum() / (len(logits) - 1)) ** 0.5
            logits = logits / div
    
            probabilities = np.exp(beta * logits)
            norm = probabilities.sum()
    
            if norm == np.infty:
                action_idx = np.argmax(logits)
                action = np.arange(len(out))[available][action_idx]
                return action, 1.0
    
            probabilities /= norm
        else:
            probabilities = [1]
    
        action_idx = np.random.choice(np.arange(len(probabilities)), p=probabilities)
        action = np.arange(len(out))[available][action_idx]
        return action

    def soft_select_action(self, graph, available, beta):
        batched_graph = dgl.batch([graph, ])
        available = available.astype(bool)
        out = self.brain.predict(batched_graph, target=False)[0].reshape(-1)

        if len(out[available]) > 1:
            logits = out[available] - out[available].mean()
            div = ((logits ** 2).sum() / max((len(logits) - 1), 1)) ** 0.5
            if div > 0:
                logits = logits / div

            logits -= logits.max()
            # probabilities = np.exp(beta * logits)
            probabilities = np.exp(logits)
            norm = probabilities.sum()

            if norm == 0 or np.isinf(norm):
                action_idx = np.argmax(logits)
                action = np.arange(len(out))[available][action_idx]
                return action, 1.0

            probabilities /= max(norm, 1e-8)  # 防止除以0
        else:
            probabilities = [1]



        action_idx = np.random.choice(np.arange(len(probabilities)), p=probabilities)
        action = np.arange(len(out))[available][action_idx]
        return action

    def get_targets(self, batch):
        """
        Compute the TD-errors using the n-step Q-learning function and the model prediction
        :param batch: the batch to process
        :return: the state input, the true y, and the error for updating the memory replay
        """

        batch_len = len(batch)
        graph, avail = list(zip(*[e[1][0] for e in batch]))
        next_graph, next_avail = list(zip(*[e[1][3] for e in batch]))
        # #------增加graph的GPU的转换
        # if self.args.mode == 'gpu' and torch.cuda.is_available():
        #     graph = [g.to('cuda') for g in graph]

        if self.args.mode == 'gpu' and torch.cuda.is_available():#后面没用到的话就会自己回收了
            graph = [g.to('cuda') for g in graph]
            next_graph = [g.to('cuda') for g in next_graph]
        
        graph_batch = dgl.batch(graph)        
        next_graph_batch = dgl.batch(next_graph)
        #------增加graph的GPU的转换

        next_copy_graph_batch = dgl.batch(dgl.unbatch(next_graph_batch))
        p = self.brain.predict(graph_batch, target=False)
        #print(f"[get_targets]- next_graph.device: {next_graph[0].device}-next_copy_graph_batch:{next_copy_graph_batch.device}") #[get_targets]- next_graph.device: cuda:0-next_copy_graph_batch:cuda:0
        
        

        if next_graph_batch.number_of_nodes() > 0:
            p_ = self.brain.predict(next_graph_batch, target=False)
            p_target_ = self.brain.predict(next_copy_graph_batch, target=True)
            # print("p_", p_)
            # print("p_target_", p_target_)


        x = []
        y = []
        errors = np.zeros(len(batch))

        for i in range(batch_len):

            sample = batch[i][1]
            state_graph, state_avail = sample[0]
            action = sample[1]
            reward = sample[2]
            next_state_graph, next_state_avail = sample[3]
            next_action_indices = np.argwhere(next_state_avail == 1).reshape(-1)
            t = p[i]

            q_value_prediction = t[action]

            if len(next_action_indices) == 0:

                td_q_value = reward
                t[action] = td_q_value

            else:

                # mask = np.zeros(p_[i].shape, dtype=bool)
                mask = np.zeros(p_target_[i].shape, dtype=bool)
                mask[next_action_indices] = True

                # best_valid_next_action_id = np.argmax(p_[i][mask])
                best_valid_next_action_id = np.argmax(p_[i][mask])
                best_valid_next_action = np.arange(len(mask))[mask.reshape(-1)][best_valid_next_action_id]

                td_q_value = reward + GAMMA * p_target_[i][best_valid_next_action]
                t[action] = td_q_value

            state = (state_graph, state_avail)
            x.append(state)
            y.append(t)

            errors[i] = abs(q_value_prediction - td_q_value)

        return x, y, errors

    def learning(self):
        """
        execute a learning step on a batch of randomly selected experiences from the memory
        :return: the subsequent loss
        """

        batch = self.memory.sample(self.args.batch_size)

        x, y, errors = self.get_targets(batch)#这个函数里面已经用gpu加速了的

        #  update the errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])#这个函数里面已经用gpu加速了的
        
        loss = self.brain.train(x, y)

        # print("--- learn_loss:", loss, "---")
        return round(loss, 4)

    def plot_and_save(self, x, y, x_label, y_label, title, save_dir, file_name, show_plot=False):

        save_dir = f"{save_dir}/task_{self.args.n_task}_device_{self.args.n_device}_h_max_{self.args.n_h_max}_multi_{self.args.num_of_process}"
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
        #print(f"Plot saved to {out_fig_file}")

        if show_plot:
            plt.show()

        # 清理图形资源
        plt.clf()
        plt.close()

        out_csv_file = '%s/%s.csv' % (save_dir, title)#out_csv_file = f"{save_dir}/{title}.csv"

        with open(out_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([x_label, title])
            for i, j in zip(x, y):
                writer.writerow([i, j])

    def only_save(self, x, y, x_label, y_label, title, save_dir, file_name, show_plot=False):

        save_dir = f"{save_dir}/task_{self.args.n_task}_device_{self.args.n_device}_h_max_{self.args.n_h_max}_multi_{self.args.num_of_process}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

             
        out_csv_file = '%s/%s.csv' % (save_dir, title)#out_csv_file = f"{save_dir}/{title}.csv"

        with open(out_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([x_label, title])
            for i, j in zip(x, y):
                writer.writerow([i, j])





#=================================类外的方法========================================
    @staticmethod
    def worker_run_env(queue, env_seed, process_id,
                       initial_env, initial_brain,
                       #instance,
                       temperature,UNDERSAMPLE_RATE,args,episode_idx, memory_initialization, is_greedy):#我们没有使用self传入？为了避免几个进程之间可能会争夺self的参数以及函数
            """
            每个进程的独立运行函数
            :param queue: 共享队列
            :param env_seed: 当前进程的随机种子
            :param initial_state: 初始环境状态（深拷贝）
            :param args: 超参数
            :param n_step: 最大步数
            :param reward_threshold: 奖励阈值（undersample 逻辑）
            :param batch_size: 批量大小
            """
            # 设置当前进程的随机种子
            
            #instance = DAGGenerator(args.n_task, args.n_edge, args.n_h_max, args.k_bound, args.n_device, args.seed)
            #torch.set_num_threads(1)
            
            #env = Environment(instance, reward_scaling, args.n_step, args)

            np.random.seed(env_seed)
            env = copy.deepcopy(initial_env)  # 假设 initial_state 是 Environment 实例
            brain=copy.deepcopy(initial_brain)  

                         
            def soft_select_action(graph, available, beta):
                batched_graph = dgl.batch([graph, ])
                available = available.astype(bool)
                out = brain.predict(batched_graph, target=False)[0].reshape(-1)

                if len(out[available]) > 1:
                    logits = out[available] - out[available].mean()
                    div = ((logits ** 2).sum() / max((len(logits) - 1), 1)) ** 0.5
                    if div > 0:
                        logits = logits / div

                    logits -= logits.max()
                    # probabilities = np.exp(beta * logits)
                    probabilities = np.exp(logits)
                    norm = probabilities.sum()

                    if norm == 0 or np.isinf(norm):
                        action_idx = np.argmax(logits)
                        action = np.arange(len(out))[available][action_idx]
                        return action, 1.0

                    probabilities /= max(norm, 1e-8)  # 防止除以0
                else:
                    probabilities = [1]

                action_idx = np.random.choice(np.arange(len(probabilities)), p=probabilities)
                action = np.arange(len(out))[available][action_idx]
                return action


            

            n_step=args.n_step
            reward_threshold=args.reward_threshold
            process_batch_size=args.process_batch_size
            #temperature=temperature
            #UNDERSAMPLE_RATE=UNDERSAMPLE_RATE
            # 深拷贝环境实例（确保初始状态一致）

            cur_state = env.get_initial_environment()
            
            #print(f"[INFO-Process ID : {process_id}]: PID: {os.getpid()},initial_env:{id(initial_env)} -Env :{id(env)}-env_seed :{env_seed}-cur_state :{id(cur_state)}-hash :{hash(cur_state)}") 
            
            # 初始化经验收集变量
            graph_list = []
            rewards_vector = np.zeros(n_step)
            actions_vector = np.zeros(n_step, dtype=np.int16)
            available_vector = np.zeros((n_step, env.p_num + env.t_num))
            computation_energy_vector = np.zeros((n_step, args.n_device))
            communication_energy_vector = np.zeros((n_step, args.n_device))

            idx = 0
            positive_sample = 0
            negative_sample = 0
            total_reward = 0
            total_finished_job = 0
            total_computation_energy = 0
            total_communication_energy = 0
            num_batches_package = 0

            total_package_time=0
            total_send_time=0

            #  the current temperature for the softmax selection: increase from 0 to MAX_BETA
            # temperature = max(0., min(args.max_softmax_beta,(episode_idx - 1) / STEP_EPSILON * args.max_softmax_beta))
            # UNDERSAMPLE_RATE = random.uniform(args.undersample_lower_bound, args.undersample_upper_bound)#[1/5]

            #print(f"[INFO-Process ID : {process_id}]:Start While True") 

            # ===================== 2.环境交互循环 =====================
            int_temp=0  
           
            while True:
                
                int_temp=int_temp+1           
                if int_temp % 200 == 0:            
                    print(f"[INFO-Process ID : {process_id} - {os.getpid()}] while times= {int_temp},idx = {idx}, time_slot = {cur_state.cur_time}, neg={negative_sample}, pos={positive_sample}")
                if int_temp == 1 and memory_initialization==False: 
                    print(f"[INFO-Process ID : {process_id} - {os.getpid()}] run_episode start with [DQN]: episode_idx = {episode_idx} , UNDERSAMPLE_RATE = {UNDERSAMPLE_RATE}")
                if int_temp == 1 and memory_initialization==True: 
                    print(f"[INFO-Process ID : {process_id} - {os.getpid()}] run_episode start with [Random]: episode_idx = {episode_idx} , UNDERSAMPLE_RATE = {UNDERSAMPLE_RATE}")  
                
               
                graph = env.make_nn_input(cur_state, args.mode)
                
                avail, avail_idx = env.get_valid_actions(cur_state) 

                # if memory_initialization==False:
                #     print(f"[INFO-Process ID : {process_id} - {os.getpid()}] avail_idx  hash is {hash (tuple(avail_idx))}!!")  

                # 获取可用动作
                if avail_idx.size != 0:
                    #print(f"[INFO-Process ID : {process_id}]:if avail_idx.size{avail_idx.size} != 0:  ") 
                    if memory_initialization:  # if we are in the memory initialization phase, a random episode is selected
                        if is_greedy:
                            action = random.choice(avail_idx)
                            #action = avail_idx[np.argmax(Q_values)]）
                        else:                    
                            action = random.choice(avail_idx)
                    else:  # otherwise, we do the softmax selection
                        
                        action =soft_select_action(graph, avail, temperature)                    
                else:
                    action = 0

                # 执行动作并获取新状态
                    
                cur_state, reward, finished_job, computation_energy_state, communication_energy_state = env.get_next_state_with_reward(cur_state, action)

                if reward==0:
                    print(f"[INFO-Process ID : {process_id} - {os.getpid()}] woker run env- get_next_state_with_reward-Rewward is 0!!!")  

                # 首先正负样本判断
                if reward > reward_threshold:
                    positive_sample += 1
                else:
                    negative_sample += 1

                #这个更新频率太低了，需要最后几轮才会更新。改成每执行100次就判断正负样本
                # 每100步检查一次样本平衡（新增）
                if idx % 20 == 0 and args.undersample:
                # 计算当前正负样本比例                            
                    dynamic_ratio = (positive_sample + 1) / (negative_sample + 1)  # 拉普拉斯平滑        
                    # 改进的平衡条件（考虑动态比例和当前奖励）
                    if (dynamic_ratio < UNDERSAMPLE_RATE and  # 实际比例低于目标
                        reward < args.reward_threshold and  # 当前是负样本
                        negative_sample > positive_sample):  # 负样本确实更多              
                        if random.random() < (1 - dynamic_ratio/UNDERSAMPLE_RATE):#避免"一刀切"式的跳过，更平滑的过渡
                            negative_sample -= 1
                            continue #牺牲这个样本          

                # 更新有效经验数据
                #g = g.to(tensor_device)
                graph = env.make_nn_input(cur_state, args.mode)

                
                graph_list.append(graph)
                rewards_vector[idx] = reward
                actions_vector[idx] = action
                available_vector[idx] = avail
                #computation_energy_vector[idx] = computation_energy_state
                #communication_energy_vector[idx] = communication_energy_state

                # 统计奖励和能量消耗
                total_reward += reward
                total_finished_job += finished_job
                total_computation_energy += np.sum(computation_energy_state)
                total_communication_energy += np.sum(communication_energy_state)

                #注意computation_energy_consumption = np.zeros(self.instance.N)  # 新增：是记录每个设备的总能耗，所以需要一次np.sum
                                
                
                if (idx>1) and (idx % process_batch_size==0):
                    # 收集经验并放入队列（每 process_batch_size=64 条数据一组）
                    # 将经验数据放入队列
                    # 这里主要传输的是   learning 和加入经验池   所需要的参数
                    # 满足sample = (state_features, action, reward, next_state_features)
                    start_send_time=time.time()
                    start_package_time=time.time()

                    sample_package=[]                
                    for j in range(process_batch_size - 1):
                        #开始打包
                        i=num_batches_package+j  
                        cur_graph = graph_list[i]
                        cur_available = available_vector[i]
                        next_graph = graph_list[i + 1]
                        next_available = available_vector[i + 1]
                        #  a state correspond to the graph
                        state_features = (cur_graph, cur_available)
                        next_state_features = (next_graph, next_available)
                        #  the n-step reward
                        reward = rewards_vector[i]
                        action = actions_vector[i]
                        # print("******************************************++++++graph++++++***************************************************")
                        # print(cur_graph)
                        # print("*****************************************++++++graph结束++++++****************************************************")
                        sample = (state_features, action, reward, next_state_features)
                        sample_package.append(sample)
                    
                    
                    end_package_time=time.time()
                    
                    queue.put({
                    "type": "data",
                    "process_id":process_id,
                    "process_batch_size":process_batch_size,
                    "num_batches_package":num_batches_package,
                    "sample_package":sample_package
                    })
                    end_send_time=time.time()
                    total_package_time=total_package_time+end_package_time-start_package_time
                    total_send_time=total_send_time+end_send_time-start_send_time

                    num_batches_package=num_batches_package+1

                    print(f"[{process_id}-data-{num_batches_package}]----- sending -------total_send_time:{total_send_time},total_package_time:{total_package_time}")

                    

                idx += 1

                if idx >= n_step or cur_state.cur_time >= args.n_time_slot:
                    break           
                    
        # 所有任务完成后，发送统计信息
        # 原有的return是：return loss, temperature, total_reward, total_finished_job, mean_computation_energy, mean_communication_energy
        
            queue.put({
                "type": "end",
                "process_id":process_id,
                "temperature":temperature,
                "total_reward":total_reward,               
                "total_finished_job": total_finished_job,
                "total_computation_energy": total_computation_energy,
                "total_communication_energy": total_communication_energy
                })
            print(f"[{process_id}-End] This process is over and waiting  queue.join()")
            queue.join()
            print(f"[{process_id}-End] queue.join() is over,this process is dead")
