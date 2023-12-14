import os
import argparse
import shutil
from datetime import datetime


def parse_folder_name(folder_name):
    # 尝试解析文件夹名称中的日期时间部分
    try:
        date_str = folder_name.split('_')[0]
        return datetime.strptime(date_str, '%Y%m%d%H%M%S')
    except ValueError:
        return None


def main(start, end):
    # 遍历./exp/下的所有文件夹
    for folder in os.listdir('./exp/'):
        folder_path = os.path.join('./exp/', folder)
        if os.path.isdir(folder_path):
            folder_date = parse_folder_name(folder)
            if folder_date and start <= folder_date <= end:
                # 删除在时间段内的文件夹
                print(f"Recursively deleting {folder_path}")
                shutil.rmtree(folder_path)


parser = argparse.ArgumentParser()
parser.add_argument("--start", type=str, required=True, default=None,
                    help="Start datetime in YYYYMMDDHHMMSS format")
parser.add_argument("--end", type=str, required=True, default=None,
                    help="End datetime in YYYYMMDDHHMMSS format")

args = parser.parse_args()

if args.start is None or args.end is None:
    raise ValueError("Start and end datetimes must be specified")

# 将命令行参数从字符串转换为datetime对象
start_date = datetime.strptime(args.start, '%Y%m%d%H%M%S')
end_date = datetime.strptime(args.end, '%Y%m%d%H%M%S')

main(start_date, end_date)
