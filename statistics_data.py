import os
import pandas as pd

# 定义目录和关键字
exp_dir = 'exp'
keys_to_extract = ['id', 'time', 'success',
                   'problem', 'number of objects', 'number of variables',
                   'algorithm choice', 'phase list', 'phase strategy', 'first phase rate',
                   'population size', 'maximum number of function evaluations',
                   'igd', 'gd', 'hv', 'time cost']

# 检查Excel文件是否存在，并读取已有数据（如果存在）
excel_file = 'data.xlsx'
# if os.path.exists(excel_file):
#     existing_df = pd.read_excel(excel_file)
# else:
existing_df = pd.DataFrame(columns=['Folder'] + keys_to_extract)

# 遍历exp文件夹
for folder in os.listdir(exp_dir):
    folder_path = os.path.join(exp_dir, folder)
    if os.path.isdir(folder_path):
        result_file = os.path.join(folder_path, 'result.txt')
        if os.path.exists(result_file):
            # 初始化字典，确保所有键都有一个空字符串作为默认值
            result_data = {key: '' for key in keys_to_extract}
            with open(result_file, 'r') as file:
                for line in file:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        if key in keys_to_extract:
                            result_data[key] = value.strip()
            # 添加文件夹名称作为首列
            result_data['Folder'] = folder
            # 将结果添加到现有DataFrame中
            existing_df = pd.concat([existing_df, pd.DataFrame([result_data])], ignore_index=True)
            # existing_df = existing_df.append(result_data, ignore_index=True)

# 保存到Excel文件和CSV文件（如果文件已存在，则覆盖）
existing_df.to_excel(excel_file, index=False)
csv_file = 'data.csv'
existing_df.to_csv(csv_file, index=False)
print(f"Results have been saved to {excel_file} and {csv_file}")

