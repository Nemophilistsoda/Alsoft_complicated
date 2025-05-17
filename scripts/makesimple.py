# 做一个简单的目录构建和文件生成的工具
import os  # 导入os模块，用于处理文件和目录
import sys  # 导入sys模块，用于获取命令行参数
import argparse  # 导入argparse模块，用于解析命令行参数

def create_directory_structure(root_dir, sub_dirs):
    for sub_dir in sub_dirs:  # 遍历子目录列表
        sub_dir_path = os.path.join(root_dir, sub_dir)  # 构建子目录的完整路径
        if not os.path.exists(sub_dir_path):  # 如果子目录不存在
            os.makedirs(sub_dir_path)  # 创建子目录
            print(f"Created directory: {sub_dir_path}")  # 打印创建的目录路径

def generate_files(root_dir, sub_dirs, num_files):
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(root_dir, sub_dir)
        for i in range(1, num_files + 1):
            file_path = os.path.join(sub_dir_path, f"file{i}.txt")
            if not os.path.exists(file_path):
                with open(file_path, 'w') as file:
                    file.write(f"This is file {i} in directory {sub_dir}.")
                print(f"Generated file: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create directory structure and generate files.")
    parser.add_argument("root_dir", help="Root directory to create the structure in")
    parser.add_argument("sub_dirs", nargs='+', help="Subdirectories to create")

