"""运行多个实验的脚本"""
import os
import itertools
import datetime
import random
import subprocess
from tqdm import tqdm

# 实验主题
DESC = "输出算法步骤的时间"
start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 定义不同的参数值
# problems = ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7']
# n_objs = [2, 3]
# n_vars = [2, 3, 5]
# algs = [0, 4]
# phase_lists = [0, 1]
# rates = [0.2, 0.4, 0.6, 0.8]
# strategies = [2, 3, 4]
# max_fes = [250]

# problems = ['dtlz7']
# n_objs = [3]
# n_vars = [3]
# algs = [4]
# phase_lists = [0]
# rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# strategies = [2]
# max_fes = [250]

problems = ['dtlz7']
n_objs = [3]
n_vars = [3]
algs = [4]
phase_lists = [0]
rates = [0.5]
strategies = [2]
max_fes = [250]

# 创建exp和log文件夹
os.makedirs("./exp", exist_ok=True)

with open("./log.txt", "a", encoding="utf-8") as log_file:
    log_file.write(
        f"{start_time} [Experiment: {DESC}] ##################\n")

total_combinations = len(list(itertools.product(
    problems, n_objs, n_vars, algs, phase_lists, rates, strategies, max_fes)))

# 使用 itertools.product 生成所有可能的参数组合
with tqdm(total=total_combinations, desc="Processing", unit="it") as pbar:
    for combination in itertools.product(problems, n_objs, n_vars, algs, phase_lists,
                                         rates, strategies, max_fes):
        # 解包组合
        problem, n_obj, n_var, alg, phase_list, rate, strategy, max_fe = combination

        # 生成ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        RANSTR = ''.join(random.choices(
            'abcdefghijklmnopqrstuvwxyz0123456789', k=5))
        id1 = f"{timestamp}_{RANSTR}"

        # 创建文件夹
        os.makedirs(f"./exp/{id1}", exist_ok=True)

        # # 构建命令字符串，模板command="${line} --id=${exp_id} > exp/${exp_id}/output.txt 2> >(tee -a log.txt >&2)"
        # output_file_path = f"./exp/{id1}/output.txt"
        # log_file_path = "./log.txt"
        # command = f"python multi_tdeadp_main_3.py --problem={problem} --n_obj={n_obj} --n_var={n_var} --alg={alg} --phase_list={phase_list} --rate={rate} --strategy={strategy} --max_fe={max_fe} --id={id1} > {output_file_path} 2> >(tee -a {log_file_path} >&2)"

        # # 打印命令
        # print(command)

        # # 执行命令
        # os.system(command)

        # pbar.update(1)

        # 构建命令字符串
        command = f"python multi_tdeadp_main_3.py --problem={problem} --n_obj={n_obj} --n_var={n_var} --alg={alg} --phase_list={phase_list} --rate={rate} --strategy={strategy} --max_fe={max_fe} --id={id1}"

        # 打印命令
        print(command)

        # 执行命令并重定向输出
        with open(f"./exp/{id1}/output.txt", "w", encoding="utf-8") as output_file, open("./log.txt", "a", encoding="utf-8") as log_file:
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"{time_str} {command}\n")
            process = subprocess.Popen(
                command, shell=True, stdout=output_file, stderr=subprocess.STDOUT)
            stdout, stderr = process.communicate()
            if stderr:
                print(stderr)
                log_file.write(stderr)

        pbar.update(1)

end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open("./log.txt", "a", encoding="utf-8") as log_file:
    log_file.write(f"{end_time} End of experiment ##################\n")
