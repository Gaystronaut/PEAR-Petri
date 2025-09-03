import sys
import os
import argparse
import setproctitle
import subprocess
from datetime import datetime
from src.DQN.trainer_dqn import TrainerDQN
#---------------
import torch
import pynvml
import psutil
import time
from threading import Timer
import multiprocessing as mp
#---------------
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Instances parameters
    parser.add_argument('--n_device', type=int, default=10) #3; :2
    parser.add_argument('--n_task', type=int, default=10)   #5; :3
    parser.add_argument('--n_edge', type=int, default=10)   #4  :2
    parser.add_argument('--n_h_max', type=int, default=2)
    parser.add_argument('--k_bound', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2024)

    # Hyper parameters
    parser.add_argument('--batch_size', type=int, default=32)
   
    parser.add_argument('--learning_rate', type=float, default=0.001) #0.000005 :0.00005
    parser.add_argument('--n_step', type=int, default=10000)#10000#一次完整的n_episode需要的步数
    parser.add_argument('--n_time_slot', type=int, default=300)#300一次完整的n_episode的时间，也就是Steps per update
    # n_step（最大步数）和 n_time_slot（最大时间槽）会相互约束，谁先达到条件，当前 episode 就会终止
    #最慢的情况是5min一个ep，大概一小时12个，十三个小时就150个
    parser.add_argument('--lamda_rate', type=float, default=1.3)
    parser.add_argument('--count_violated', default=False)
    parser.add_argument('--max_softmax_beta', type=int, default=10, help="max_softmax_beta")
    parser.add_argument('--hidden_layer', type=int, default=32)#32
    parser.add_argument('--latent_dim', type=int, default=128, help='dimension of latent layers')



    # Argument for Trainer
    parser.add_argument('--n_episode', type=int, default=10000)#10000
    parser.add_argument('--save_dir', type=str, default='./result-default')
    parser.add_argument('--plot_training', type=int, default=1)
    parser.add_argument('--mode', default='gpu', help='cpu/gpu')
    parser.add_argument('--use_random', default=True, help='whether to use random choice instead of Q network')
    parser.add_argument('--use_greedy', default=False, help='whether to deploy greedy strategy instead of random')
    parser.add_argument('--load_bool', default=False, help='whether to load a trained model')
    parser.add_argument('--load_memory_bool', default=False, help='whether to load the memory')
    parser.add_argument('--saved_model_name', default='iter_900_model.pth.tar', help='model file name')
    parser.add_argument('--undersample', default=False)
    parser.add_argument('--undersample_upper_bound',type=float, default=1/5, help="[0,1]")
    parser.add_argument('--undersample_lower_bound',type=float, default=1/10, help="[0,1]")
    parser.add_argument('--reward_threshold',type=int, default=5)
    parser.add_argument('--save_printf_txt', default=True, help='whether to save all printf into txt ')
    parser.add_argument('--save_dir_initiliaze_memory_data', type=str, help='where to save initiliazed memory ')

    # Argument for Multiprocess
    parser.add_argument('--process_batch_size', type=int, default=100,help="the number of each process send by once")
    parser.add_argument('--num_of_process', type=int, default=6,help="the number of process created")
    parser.add_argument('--learing_frequence', type=int, default=64,help="the number of sample used by every learing()")

    
    

    return parser.parse_args()
#---------------信息展示------------------
# GPU 信息
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
pynvml.nvmlShutdown()

# CPU 信息
cpu_util = psutil.cpu_percent(interval=0.1)
cpu_cores = psutil.cpu_count(logical=False)

def get_hardware_stats():
    """获取并打印 GPU 和 CPU 状态"""
    if not torch.cuda.is_available():
        print("CUDA 不可用，请检查 GPU 驱动和 PyTorch 版本！")
        return    
    # 打印结果
    print(f"\n[状态统计 {time.strftime('%Y-%m-%d %H:%M:%S')}]:GPU 利用率: {util.gpu}% | 显存占用: {mem_info.used / mem_info.total * 100:.1f}% CPU 利用率: {cpu_util}% ")

def schedule_stats(interval=900):
    """定时触发统计（不阻塞主线程）"""
    get_hardware_stats()
    Timer(interval, schedule_stats, [interval]).start()  # 递归调用

# 全局文件句柄和原始stdout备份
original_stdout = sys.stdout
log_file_handle = None

# 获取当前时间
def get_time_str():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

class TimestampTee:
    """带时间戳的双路输出类"""
    def __init__(self, *files):
        self.files = files
        self.current_line = ""  # 用于缓存当前行
    
    def write(self, obj):
        # 将对象转换为字符串
        text = obj if isinstance(obj, str) else str(obj)
        
        # 处理文本，添加时间戳
        for char in text:
            if char == '\n':
                # 行结束，添加时间戳并输出
                self._flush_line()
            else:
                # 添加到当前行
                self.current_line += char
    
    def _flush_line(self):
        """输出当前行并添加时间戳"""
        if self.current_line:
            timestamp = get_time_str()
            full_line = f"{timestamp} {self.current_line}\n"
            
            for f in self.files:
                if hasattr(f, 'write') and not (hasattr(f, 'closed') and f.closed):
                    f.write(full_line)
                    f.flush()
            
            self.current_line = ""
    
    def flush(self):
        """刷新缓冲区"""
        if self.current_line:
            self._flush_line()
        for f in self.files:
            if hasattr(f, 'flush') and not (hasattr(f, 'closed') and f.closed):
                f.flush()

def setup_printf_redirection(args):
    """设置printf重定向到文件"""
    global log_file_handle, original_stdout
    
    # 检查是否需要保存printf输出
    if not getattr(args, 'save_printf_txt', True):
        return
    
    # 构建路径
    save_dir2 = f"{args.save_dir}/task_{args.n_task}_device_{args.n_device}_h_max_{args.n_h_max}_multi_{args.num_of_process}"
    os.makedirs(save_dir2, exist_ok=True)
    log_file_path = os.path.join(save_dir2, f"task_{args.n_task}_device_{args.n_device}_h_max_{args.n_h_max}_time_slot_{args.n_time_slot}_n_step_{args.n_step}_use_Random_{args.use_random}_multi_{args.num_of_process}.txt")
    
    # 备份原始stdout
    original_stdout = sys.stdout
    
    # 打开日志文件
    log_file_handle = open(log_file_path, 'w')
    
    # 创建带时间戳的Tee对象
    sys.stdout = TimestampTee(sys.__stdout__, log_file_handle)
    
    # 使用普通print输出初始信息（会自动添加时间戳）
    print("[INFO] Output redirection initialized")

def close_printf_redirection():
    """关闭文件并恢复原始stdout"""
    global log_file_handle, original_stdout
    
    if log_file_handle is not None:
        # 确保所有缓冲数据写入
        sys.stdout.flush()
        
        # 恢复原始stdout
        sys.stdout = original_stdout
        
        # 关闭文件
        log_file_handle.close()
        log_file_handle = None
        print("[INFO] Log file closed successfully")

if __name__ == '__main__':

    
    mp.set_start_method('spawn')

    args = parse_arguments()
    args.use_random = False
    args.undersample = True
    args.plot_training=1
    args.n_h_max=2
    args.n_task = 10
    args.n_device = 4#10
    args.batch_size = 512#64 
    args.learing_frequence = 256#64 注意不同的进程数量和device的设置，会让主线程和子进程的速度不统一
    args.hidden_layer =8
    args.load_memory_bool=False    
    args.n_step =5000#4500#10000 #6000
    args.n_time_slot =300 
    args.num_of_process =2#4 #注意显存占用！
    args.process_batch_size =200
    
    schedule_stats(interval=500)
    
    cudaset=0
    os.environ["CUDA_VISIBLE_DEVICES"] =str(cudaset) 
    setproctitle.setproctitle(f"Cuda:{cudaset}_task=:{args.n_task}_D={args.n_device}_h=:{args.n_h_max}_rand={args.use_random}_frequence={args.learing_frequence}_num_of_process={args.num_of_process}")
    
    args.save_dir='.username/D2D/username_dqn_training_results'
    args.save_dir_initiliaze_memory_data='.username/D2D/username_dqn_training_results/initialized_memory'

    # 设置printf重定向
    setup_printf_redirection(args)        

    
    print(f"*******Task: {args.n_task}, Device: {args.n_device}, h_max: {args.n_h_max},use_random:{args.use_random},_time_slot_{args.n_time_slot}*********")
               
    print(f"[INFO] save_dir: {args.save_dir}")
    print("***********************************************************")
    print("[INFO] TRAINING  Environment")
    print("[INFO] n_device: %d" % args.n_device)
    print("[INFO] n_task: %d" % args.n_task)
    print("[INFO] n_edge: %d" % args.n_edge)
    print("[INFO] n_h_max: %d" % args.n_h_max)
    print("[INFO] k_bound: %d" % args.k_bound)
    print("[INFO] seed: %s" % args.seed)
    print("***********************************************************")
    print("[INFO] TRAINING PARAMETERS")
    print("[INFO] algorithm: DQN")
    print("[INFO] batch_size: %d" % args.batch_size)
    print("[INFO] learning_rate: %f" % args.learning_rate)
    print("[INFO] hidden_layer: %d" % args.hidden_layer)
    print("[INFO] latent_dim: %d" % args.latent_dim)
    print("[INFO] softmax_beta: %d" % args.max_softmax_beta)
    print("[INFO] n_step: %d" % args.n_step)
    print("[INFO] n_time_slot: %d" % args.n_time_slot)   
    print("[INFO] n_episode: %d" % args.n_episode) 
    print("[INFO] use_random: %d" % args.use_random) 
    print("[INFO] undersample: %d" % args.undersample) 
    print("***********************************************************")
    

    sys.stdout.flush()    
    trainer = TrainerDQN(args)
    trainer.run_training()



