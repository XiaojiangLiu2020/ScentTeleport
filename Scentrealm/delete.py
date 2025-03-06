import os

def delete_files_by_extension(directory, extensions):
    """
    删除指定目录及其子目录下的指定后缀文件。

    Args:
        directory (str): 要操作的根目录路径。
        extensions (list): 文件扩展名列表（如 ['.THM', '.LRV']）。
    """
    if not os.path.exists(directory):
        print(f"目录 '{directory}' 不存在，请检查路径。")
        return

    deleted_files = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                    deleted_files += 1
                except Exception as e:
                    print(f"无法删除文件: {file_path}，错误: {e}")

    if deleted_files == 0:
        print("没有找到符合条件的文件。")
    else:
        print(f"总计删除了 {deleted_files} 个文件。")

if __name__ == "__main__":
    # 指定目标目录
    target_directory = r"D:\vlog_video"

    # 文件后缀列表
    extensions_to_delete = [".THM", ".LRV"]

    # 执行删除操作
    delete_files_by_extension(target_directory, extensions_to_delete)