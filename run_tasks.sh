#!/bin/bash

# 输出开始
echo "Start..."

# 检查并创建 exp 文件夹（如果不存在）
if [[ ! -d "exp" ]]; then
    mkdir -p "exp"
fi

# 循环读取 task.txt 中的每一行
while IFS= read -r line; do
    # 检查行是否为空
    if [[ -n "$line" ]]; then

        # 获取当前日期时间，格式为 yyyy-MM-dd-HH-mm-ss
        current_date_time=$(date +"%Y%m%d%H%M%S")
        current_data_time_fmt=$(date +"%Y-%m-%d %H:%M:%S")

        # 生成一个随机数
        random_number=$RANDOM

        # 创建新的文件夹名称，结合当前日期时间和随机数
        exp_id="${current_date_time}_${random_number}"

        # 创建文件夹
        mkdir -p "exp/$exp_id"

        # 构造命令，再line后加上--id=$exp_id
        command="${line} --id=${exp_id} > exp/${exp_id}/output.txt 2> >(tee -a log.txt >&2)"

        # 输出命令
        echo "$current_data_time_fmt $command"

        # 如果文件夹创建成功，则执行命令并重定向输出到该文件夹的 output.txt
        if [[ -d "exp/$exp_id" ]]; then
            # 将执行时间、完整命令、id记录到日志log.txt
            echo "$current_data_time_fmt $command" >> "log.txt"
            eval "$command"
        else
            echo "Failed to create directory exp/${exp_id}"
            exit 1
        fi
    fi
done < "task.txt"

# 输出结束
echo "Done."
