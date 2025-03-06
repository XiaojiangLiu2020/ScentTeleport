import asyncio
import json
import numpy as np
import matplotlib.pyplot as plt
from bleak import BleakClient, BleakScanner
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import simpledialog, filedialog
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from matplotlib import rcParams
from binascii import unhexlify
from crcmod import crcmod
import sys
from tkinter import scrolledtext
from sklearn.preprocessing import StandardScaler  # 导入标准化工具

# 设置中文字体为 SimHei（黑体），以支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

# ESP32 的 UUID
ESP32_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
ESP32_NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
ESP32_WRITE_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

# ScentDevice 的 UUID 和 MAC 地址
SCENT_NOTIFICATION_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
SCENT_WRITE_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
SCENT_MAC_ADDR = "08:D1:F9:12:D5:36"

# 气味通道映射
SCENT_CHANNEL_MAPPING = {"柠檬": 1, "咖啡d3": 2, "白酒d3": 3}

# 蓝牙指令辅助函数
FRAME_HEAD = 'F5'
FRAME_TAIL = '55'

# 创建独立窗口
root = tk.Tk()
root.title("传感器阵列实时数据")
root.geometry("1200x1000")




# 全局变量初始化
# 全局变量初始化
sensor_data_lists = [[[] for _ in range(6)] for _ in range(6)]  # 保存最近 100 个数据
sensor_data_archive = [[[] for _ in range(6)] for _ in range(6)]  # 保存所有数据
time_points = []
last_update_time = time.time()
received_data = ""  # 接收蓝牙数据
esp32_client = None  # 蓝牙客户端
is_reading = False  # 标志是否正在读数
fig = None
axes = None
lines = None
canvas = None
current_mode = None  # 当前绘图模式：None、"thumbnail" 或 "full"

# 新增全局变量
loaded_datasets = {}  # 存储所有已加载的气味数据集 {气味名称: 数据矩阵}
current_data = None   # 当前传感器阵列的实时数据

current_recognized_smell = None  # 当前识别出的气味
scent_client = None  # ScentDevice 蓝牙客户端


class RedirectOutput:
    """将标准输出和错误输出重定向到 Tkinter 文本框"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.text_widget.configure(state="disabled")  # 初始为只读
        self.text_widget.tag_config("stdout", foreground="black")  # 正常输出为黑色
        self.text_widget.tag_config("stderr", foreground="red")  # 错误输出为红色

    def write(self, message):
        """将信息写入文本框"""
        self.text_widget.configure(state="normal")  # 启用编辑
        self.text_widget.insert(tk.END, message, "stdout")  # 写入普通信息
        self.text_widget.see(tk.END)  # 滚动到文本框末尾
        self.text_widget.configure(state="disabled")  # 禁用编辑

    def flush(self):
        """刷新输出（必需，用于兼容性）"""
        pass


class RedirectErrorOutput:
    """将错误输出（stderr）重定向到 Tkinter 文本框"""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        """将错误信息写入文本框"""
        self.text_widget.configure(state="normal")  # 启用编辑
        self.text_widget.insert(tk.END, message, "stderr")  # 写入错误信息
        self.text_widget.see(tk.END)  # 滚动到文本框末尾
        self.text_widget.configure(state="disabled")  # 禁用编辑

    def flush(self):
        """刷新输出（必需，用于兼容性）"""
        pass


def add_info_bar(root):
    """
    在主窗口中添加信息栏，用于显示代码中的打印信息。
    参数:
        root: 主窗口 (tk.Tk 或 tk.Toplevel)
    """


    # 创建带滚动条的文本框
    info_bar = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10, font=("Consolas", 10))
    info_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)  # 信息栏固定在窗口底部

    # 创建信息栏标题
    #info_label = tk.Label(root, text="信息栏", font=("Arial", 12), anchor="w")
    #info_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))  # 标题固定在底部上方

    # 重定向 stdout 和 stderr 到文本框
    sys.stdout = RedirectOutput(info_bar)
    sys.stderr = RedirectErrorOutput(info_bar)

    print("信息栏已初始化，所有打印信息将显示在此处。")

def create_canvas(mode="full"):
    """创建画布和初始化子图"""
    global fig, axes, lines, canvas, current_mode

    # 清除之前的画布
    if canvas is not None:
        canvas.get_tk_widget().destroy()

    # 更新当前模式
    current_mode = mode

    if mode == "full":
        # 创建 6x6 子图
        fig, axes = plt.subplots(6, 6, figsize=(12, 12))
        fig.tight_layout()
        lines = [[None for _ in range(6)] for _ in range(6)]

        # 初始化每个子图
        for i in range(6):
            for j in range(6):
                ax = axes[i][j]
                lines[i][j], = ax.plot([], [], label=f"Sensor ({i},{j})", color="blue")
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100000)
                ax.set_title(f"Sensor ({i},{j})", fontsize=8)
                ax.grid(True)
                ax.tick_params(labelsize=6)
                ax.legend(fontsize=6)
    elif mode == "thumbnail":
        # 创建 2x2 子图，仅绘制 (2,2)、(2,3)、(3,2)、(3,3)
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.tight_layout()
        lines = [[None for _ in range(6)] for _ in range(6)]

        # 定义缩略图的子图映射
        thumbnail_positions = [(2, 2), (2, 3), (3, 2), (3, 3)]
        for idx, (i, j) in enumerate(thumbnail_positions):
            ax = axes[idx // 2][idx % 2]
            lines[i][j], = ax.plot([], [], label=f"Sensor ({i},{j})", color="blue")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100000)
            ax.set_title(f"Sensor ({i},{j})", fontsize=8)
            ax.grid(True)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6)

    # 将 matplotlib 图形嵌入 tkinter 窗口
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def update_plot():
    """优化后的更新实时数据的绘图"""
    global last_update_time, current_mode, canvas, time_points
    if canvas is None:  # 如果画布未初始化，直接返回
        return

    current_time = time.time()

    try:
        if current_mode == "full":
            # 限制更新频率为每 1 秒更新一次
            update_interval = 1  # 更新间隔，单位为秒
            if current_time - last_update_time < update_interval:
                return
            last_update_time = current_time

            # 更新所有子图
            for i in range(6):
                for j in range(6):
                    y_data = sensor_data_lists[i][j]
                    if not y_data:  # 如果没有数据，跳过
                        continue

                    # 动态更新横轴范围
                    if len(y_data) <= 100:
                        x_data = list(range(len(y_data)))  # 使用完整的索引
                    else:
                        x_data = list(range(len(y_data) - 100, len(y_data)))  # 仅保留最近 100 个索引
                        y_data = y_data[-100:]  # 仅保留最近 100 个数据点

                    # 更新子图
                    lines[i][j].set_data(x_data, y_data)

                    # 动态调整坐标轴范围
                    if y_data:
                        axes[i][j].set_ylim(min(y_data) - 1000, max(y_data) + 1000)
                        axes[i][j].set_xlim(min(x_data), max(x_data))  # 根据实际索引更新横轴范围

        elif current_mode == "thumbnail":
            # 实时更新缩略图 (2,2)、(2,3)、(3,2)、(3,3)
            thumbnail_positions = [(2, 2), (2, 3), (3, 2), (3, 3)]
            for idx, (i, j) in enumerate(thumbnail_positions):
                y_data = sensor_data_lists[i][j]
                if not y_data:  # 如果没有数据，跳过
                    continue

                # 动态更新横轴范围
                if len(y_data) <= 100:
                    x_data = list(range(len(y_data)))  # 使用完整的索引
                else:
                    x_data = list(range(len(y_data) - 100, len(y_data)))  # 仅保留最近 100 个索引
                    y_data = y_data[-100:]  # 仅保留最近 100 个数据点

                # 更新缩略图
                lines[i][j].set_data(x_data, y_data)

                # 动态调整坐标轴范围
                if y_data:
                    axes[idx // 2][idx % 2].set_ylim(min(y_data) - 1000, max(y_data) + 1000)
                    axes[idx // 2][idx % 2].set_xlim(min(x_data), max(x_data))  # 根据实际索引更新横轴范围

        canvas.draw()  # 更新画布
    except Exception as e:
        print(f"绘图更新错误: {e}")
#数据去极端值在handler里
def esp32_notification_handler(sender, data):
    """处理 ESP32 的蓝牙通知，并分别保存最近 100 个数据和全部数据"""
    global received_data, time_points, sensor_data_lists, sensor_data_archive

    # 设置上下限
    LOWER_LIMIT = -1
    UPPER_LIMIT = 1000000000

    try:
        # 接收数据的拼接与解析
        chunk = data.decode("utf-8").strip()
        received_data += chunk

        if "#END#" in received_data:
            complete_data, received_data = received_data.split("#END#", 1)
            sensor_matrix = json.loads(complete_data)  # 将 JSON 数据解析为矩阵
            # print(f"接收到完整数据: {complete_data}")

            if len(time_points) == 0:
                time_points = [0]  # 初始化时间点
            else:
                time_points.append(time_points[-1] + 1)  # 增加时间点

            # 对传感器数据进行上下限处理
            for i in range(6):  # 遍历传感器矩阵的每一行
                for j in range(6):  # 遍历传感器矩阵的每一列
                    current_value = sensor_matrix[i][j]  # 当前接收到的传感器值

                    # 检查当前值是否在上下限范围内
                    if LOWER_LIMIT <= current_value <= UPPER_LIMIT:
                        # 如果在范围内，正常记录
                        sensor_data_lists[i][j].append(current_value)
                        sensor_data_archive[i][j].append(current_value)  # 保存到所有数据的存储变量
                    else:
                        # 如果超出上下限范围，继承上一时间点的数据
                        if len(sensor_data_lists[i][j]) > 0:
                            # 继承上一时间点的数据
                            sensor_data_lists[i][j].append(sensor_data_lists[i][j][-1])
                            sensor_data_archive[i][j].append(sensor_data_lists[i][j][-1])
                        else:
                            # 如果没有历史数据，则记录为 0 或其他默认值
                            sensor_data_lists[i][j].append(0)
                            sensor_data_archive[i][j].append(0)

                    # 限制每个传感器数据的历史长度（保留最近 100 个数据点）
                    if len(sensor_data_lists[i][j]) > 100:
                        sensor_data_lists[i][j] = sensor_data_lists[i][j][-100:]

    except json.JSONDecodeError as e:
        print(f"解析数据时出错: {e}")
    except Exception as e:
        print(f"其他错误: {e}")


async def connect_bluetooth():
    """连接蓝牙设备"""
    global esp32_client
    print("正在扫描 ESP32...")
    device = await BleakScanner.find_device_by_name("ESP32BLE")
    if not device:
        print("未找到 ESP32 设备")
        return

    esp32_client = BleakClient(device)
    await esp32_client.connect()
    print("蓝牙已连接")


async def disconnect_bluetooth():
    """断开蓝牙设备"""
    global esp32_client, is_reading
    if esp32_client and esp32_client.is_connected:
        await esp32_client.disconnect()
        print("蓝牙已断开")
    esp32_client = None
    is_reading = False


async def start_reading():
    """开始读数"""
    global esp32_client, is_reading
    if not esp32_client or not esp32_client.is_connected:
        print("请先连接蓝牙设备")
        return

    await esp32_client.start_notify(ESP32_NOTIFY_UUID, esp32_notification_handler)
    print("已订阅通知")
    await esp32_client.write_gatt_char(ESP32_WRITE_UUID, "read_sensor_array".encode())
    print("已发送 read_sensor_array 指令")
    print("正在读数...")
    is_reading = True


async def stop_reading():
    """停止读数"""
    global esp32_client, is_reading
    if not esp32_client or not esp32_client.is_connected:
        print("蓝牙未连接，无法发送停止指令")
        return

    try:
        await esp32_client.stop_notify(ESP32_NOTIFY_UUID)
        print("已成功停止蓝牙通知")
    except Exception as e:
        print(f"停止蓝牙通知时发生错误: {e}")
        return

    try:
        await esp32_client.write_gatt_char(ESP32_WRITE_UUID, "stop_sensor_array".encode())
        print("已发送停止指令")
    except Exception as e:
        print(f"发送停止指令时发生错误: {e}")
        return

    is_reading = False


def save_all_sensor_data():
    """将 sensor_data_archive 保存为 .xlsx 文件"""
    global sensor_data_archive

    try:
        # 转换 sensor_data_archive 为 pandas DataFrame
        data = {}
        for i in range(6):
            for j in range(6):
                sensor_key = f"Sensor_{i}_{j}"
                data[sensor_key] = sensor_data_archive[i][j]  # 每个传感器的全部数据

        # 创建 DataFrame
        max_length = max(len(values) for values in data.values())  # 找到最长数据列表长度
        for key in data:
            # 补齐短数据以对齐表格（用 None 填充）
            data[key] += [None] * (max_length - len(data[key]))
        df = pd.DataFrame(data)

        # 弹出文件保存对话框
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel 文件", "*.xlsx")],
            title="保存所有传感器数据"
        )
        if file_path:
            # 保存到 Excel 文件
            df.to_excel(file_path, index=False)
            print(f"传感器数据已保存到 {file_path}")
        else:
            print("保存操作已取消。")
    except Exception as e:
        print(f"保存数据时出错: {e}")


async def start_training_mode():
    """训练模式：保存气味数据集"""
    global is_reading, sensor_data_lists

    # 停止读数
    if is_reading:
        await stop_reading()
        print("已停止读数，进入训练模式。")

    # 如果没有数据，提示用户
    if not any(sensor_data_lists[i][j] for i in range(6) for j in range(6)):
        print("没有可用数据，请先开始读数。")
        return

    # 自定义窗口
    def open_training_window():
        """打开训练模式窗口"""
        def save_data():
            """保存数据"""
            try:
                # 获取用户输入的时间点和气味名称
                selected_time_point = int(time_point_entry.get())
                smell_name = smell_name_entry.get()

                # 检查输入有效性
                max_time_point = max(len(sensor_data_lists[i][j]) for i in range(6) for j in range(6))
                if selected_time_point < 0 or selected_time_point >= max_time_point:
                    error_label.config(text=f"时间点超出范围，请输入 0 ~ {max_time_point - 1}！", fg="red")
                    return
                if not smell_name:
                    error_label.config(text="气味名称不能为空！", fg="red")
                    return

                # 提取从选择时间点开始的10个时间点的数据
                start_index = selected_time_point
                end_index = min(start_index + 3, max_time_point)  # 确保不超过数据长度

                extracted_data = []
                for i in range(6):
                    for j in range(6):
                        # 根据用户输入时间点提取对应数据
                        extracted_data.append(sensor_data_lists[i][j][start_index:end_index])

                # 转换为 DataFrame
                df = pd.DataFrame(extracted_data, index=[f"Sensor_{i}_{j}" for i in range(6) for j in range(6)])
                df.columns = [f"Time_{k}" for k in range(end_index - start_index)]

                # 保存为 Excel 文件
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel 文件", "*.xlsx")],
                    title="保存数据集",
                )
                if file_path:
                    df.to_excel(file_path, index=True)
                    print(f"数据已保存到 {file_path}，气味名称: {smell_name}")
                    error_label.config(text=f"数据已保存到 {file_path}", fg="green")
                else:
                    error_label.config(text="保存操作已取消。", fg="orange")

                # 关闭窗口
                training_window.destroy()

            except ValueError:
                error_label.config(text="时间点必须是数字！", fg="red")

        # 创建新窗口
        training_window = tk.Toplevel(root)
        training_window.title("训练模式")
        training_window.geometry("400x300")

        # 时间点输入框
        max_time_point = max(len(sensor_data_lists[i][j]) for i in range(6) for j in range(6))
        time_point_label = tk.Label(training_window, text=f"选择时间点（0 ~ {max_time_point - 1}）：")
        time_point_label.pack(pady=10)
        time_point_entry = tk.Entry(training_window)
        time_point_entry.pack(pady=5)

        # 气味名称输入框
        smell_name_label = tk.Label(training_window, text="输入气味名称：")
        smell_name_label.pack(pady=10)
        smell_name_entry = tk.Entry(training_window)
        smell_name_entry.pack(pady=5)

        # 错误提示标签
        error_label = tk.Label(training_window, text="", fg="red")
        error_label.pack(pady=10)

        # 保存按钮
        save_button = tk.Button(training_window, text="保存数据", command=save_data)
        save_button.pack(pady=20)

        # 窗口关闭按钮
        close_button = tk.Button(training_window, text="关闭", command=training_window.destroy)
        close_button.pack(pady=10)

    # 打开训练窗口
    open_training_window()


def clear_data():
    """清空传感器数据"""
    global sensor_data_lists, time_points
    sensor_data_lists = [[[] for _ in range(6)] for _ in range(6)]
    time_points = []
    print("已清空数据")


def switch_to_thumbnail():
    """切换到缩略图模式"""
    create_canvas(mode="thumbnail")


def switch_to_full_plot():
    """切换到完整绘图模式"""
    create_canvas(mode="full")


def run_async_task(coroutine):
    """将异步协程包装为任务"""
    asyncio.create_task(coroutine)

def load_datasets():
    """加载已有的气味数据集"""
    global loaded_datasets
    loaded_datasets = {}  # 清空已有数据集
    file_paths = filedialog.askopenfilenames(
        title="选择气味数据集文件",
        filetypes=[("Excel 文件", "*.xlsx")]
    )
    for file_path in file_paths:
        smell_name = file_path.split("/")[-1].split(".")[0]  # 使用文件名作为气味名称
        df = pd.read_excel(file_path, index_col=0)
        loaded_datasets[smell_name] = df.values.flatten()  # 展开为向量
    print(f"已加载数据集: {list(loaded_datasets.keys())}")


def get_current_data():
    """获取当前传感器阵列数据（最近 x 个时间点）"""
    global current_data
    if len(time_points) < 3:
        print("当前数据不足 3 个时间点，无法进行识别模式。")
        return None

    # 提取最近 x 个时间点的数据
    current_data = []
    for i in range(6):
        for j in range(6):
            current_data.extend(sensor_data_lists[i][j][-3:])
    current_data = np.array(current_data)  # 转为 NumPy 数组
    #print("当前数据已获取。")
    return current_data


async def start_recognition_mode():
    """识别模式：实时分析当前数据和已有数据集的相似性，并支持控制 ScentDevice"""
    global current_data, is_reading, loaded_datasets, current_recognized_smell, scent_client

    # 加载已有数据集
    load_datasets()
    if not loaded_datasets:
        print("未加载任何数据集，无法启动识别模式。")
        return

    # 创建识别模式窗口
    recognition_window = tk.Toplevel(root)
    recognition_window.title("实时识别模式")
    recognition_window.geometry("800x650")

    # 创建用于显示最相似气味的标签
    recognition_label = tk.Label(recognition_window, text="最相似气味：", font=("Arial", 14))
    recognition_label.pack(pady=10)

    # 创建输入框供用户设置 similarity_limit
    similarity_label = tk.Label(recognition_window, text="设置相似度阈值（similarity_limit）：", font=("Arial", 12))
    similarity_label.pack(pady=10)
    similarity_entry = tk.Entry(recognition_window)
    similarity_entry.pack(pady=5)
    similarity_entry.insert(0, "5")  # 默认值

    # 创建 matplotlib 图形容器
    fig, ax = plt.subplots(figsize=(8, 6))
    canvas = FigureCanvasTkAgg(fig, master=recognition_window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    async def update_recognition_plot():
        global current_recognized_smell
        """实时更新识别图像和识别结果"""
        nonlocal fig, ax, canvas
        while True:
            # 获取当前数据
            current_data = get_current_data()
            if current_data is None:
                recognition_label.config(text="当前数据不足，无法识别。")
                await asyncio.sleep(1)
                continue

            # 合并所有数据集和当前数据
            all_data = np.array(list(loaded_datasets.values()))
            all_labels = list(loaded_datasets.keys())
            all_data = np.vstack([all_data, current_data])  # 添加当前数据
            all_labels.append("当前数据")  # 添加当前数据的标签

            # 标准化数据
            scaler = StandardScaler()  # 创建标准化对象
            all_data = scaler.fit_transform(all_data)  # 对所有数据进行标准化

            # PCA 降维到 2 维
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(all_data)

            # 清空旧图像并绘制新图像
            ax.clear()

            # 创建一个字典，用于存储每种气味的颜色
            color_map = {}
            unique_smells = set()  # 用于保存已绘制图例的气味
            current_color_index = 0
            colors = plt.cm.tab10.colors  # 使用 matplotlib 提供的颜色映射

            # 绘制已有数据集
            for i, label in enumerate(all_labels[:-1]):
                # 提取气味名称（下划线前部分）
                smell_name = label.split("_")[0]

                # 为每种气味分配颜色
                if smell_name not in color_map:
                    color_map[smell_name] = colors[current_color_index % len(colors)]
                    current_color_index += 1

                # 绘制点
                ax.scatter(
                    reduced_data[i, 0],
                    reduced_data[i, 1],
                    color=color_map[smell_name],
                    label=smell_name if smell_name not in unique_smells else None,  # 避免重复图例
                    alpha=0.7
                )
                unique_smells.add(smell_name)

            # 绘制当前数据
            ax.scatter(
                reduced_data[-1, 0],
                reduced_data[-1, 1],
                label="当前数据",
                color="red",
                marker="*",
                s=200
            )

            # 设置图标题和轴标签
            ax.set_title("PCA 降维后气味数据分布")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")

            # 添加图例（避免重复）
            ax.legend()
            ax.grid()

            # 更新 matplotlib 图像
            canvas.draw()

            # 计算相似性（欧几里得距离）
            distances = euclidean_distances([reduced_data[-1]], reduced_data[:-1])[0]
            most_similar_index = np.argmin(distances)
            most_similar_label = all_labels[most_similar_index]
            similarity_score = distances[most_similar_index]

            # 提取最相似气味的气味名称（保留下划线前的部分）
            most_similar_smell_name = most_similar_label.split("_")[0]

            # 获取用户设置的 similarity_limit
            try:
                similarity_limit = float(similarity_entry.get())
            except ValueError:
                similarity_limit = 0.5  # 默认值
                similarity_entry.delete(0, tk.END)
                similarity_entry.insert(0, "0.5")

            # 根据阈值判断是否匹配
            if similarity_score <= similarity_limit:
                recognition_result = f"最相似气味：{most_similar_smell_name}，相似度（距离）：{similarity_score:.4f}"
                current_recognized_smell = most_similar_smell_name  # 更新全局变量

            else:
                recognition_result = f"未识别气味，距离：{similarity_score:.4f}"
                current_recognized_smell = None  # 未识别任何气味

            # 更新识别结果标签
            recognition_label.config(text=recognition_result)

            # 每秒更新一次
            await asyncio.sleep(1)

    # 启动实时更新的协程
    asyncio.create_task(update_recognition_plot())



def crc16Add(str_data):
    crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)
    data = str_data.replace(" ", "")
    readcrcout = hex(crc16(unhexlify(data))).upper()
    str_list = list(readcrcout)
    if len(str_list) < 6:
        str_list.insert(2, '0' * (6 - len(str_list)))
    crc_data = "".join(str_list)
    return crc_data[2:4] + ' ' + crc_data[4:]


def cmd2bytearray(cmd_str: str):
    verify = crc16Add(cmd_str)
    cmd = FRAME_HEAD + ' ' + cmd_str + ' ' + verify + ' ' + FRAME_TAIL
    return bytearray.fromhex(cmd)


def start_play(scent: int, playtime: int):
    play_cmd = '00 00 00 01 02 05'
    scent_channel = f"{scent:02x}"
    if playtime == 0:  # 无限播放
        playtime16 = 'FF FF FF FF'
    else:
        playtime16 = f"{playtime:08x}"
    cmd_data = f"{play_cmd} {scent_channel} {playtime16}"
    return cmd2bytearray(cmd_data)


def stop_play():
    stop_cmd = '00 00 00 01 00 01 00'
    return cmd2bytearray(stop_cmd)

# 全局变量
scentdevice_is_running = False  # 标志函数是否正在运行
scentdevice_stop_requested = False  # 标志是否请求停止
async def control_scent_device():
    """控制 ScentDevice 的气味播放，并避免重复触发播放指令"""
    global current_recognized_smell, scent_client, scentdevice_is_running, scentdevice_stop_requested

    # 如果函数已经在运行，则请求停止
    if scentdevice_is_running:
        scentdevice_stop_requested = True  # 设置停止标志
        print("正在停止 ScentDevice 控制...")
        return

    # 否则开始运行
    scentdevice_is_running = True  # 设置运行标志
    scentdevice_stop_requested = False  # 重置停止标志

    print("正在扫描 ScentDevice...")
    scent_device = await BleakScanner.find_device_by_address(SCENT_MAC_ADDR)
    if not scent_device:
        print("未找到 ScentDevice")
        scentdevice_is_running = False  # 恢复运行标志
        return

    try:
        async with BleakClient(scent_device) as client:
            scent_client = client
            print("已连接到 ScentDevice")

            last_recognized_smell = None  # 记录上一次的识别结果

            # 每秒检查识别的气味并控制设备
            while True:
                # 如果请求停止，则断开连接并退出
                if scentdevice_stop_requested:
                    print("已收到停止请求，断开连接...")
                    stop_cmd = stop_play()  # 停止播放
                    await client.write_gatt_char(SCENT_WRITE_UUID, stop_cmd)
                    break

                recognized_smell = current_recognized_smell

                # 如果当前识别的气味与上一次相同，跳过播放指令

                if recognized_smell != last_recognized_smell:
                    if recognized_smell in SCENT_CHANNEL_MAPPING:
                        scent_channel = SCENT_CHANNEL_MAPPING[recognized_smell]
                        print(f"播放气味通道: {scent_channel}（气味: {recognized_smell}）")
                        play_cmd = start_play(scent_channel, 10000)  # 播放 10 秒
                        await client.write_gatt_char(SCENT_WRITE_UUID, play_cmd)
                        last_recognized_smell = recognized_smell  # 更新上一次的识别结果
                    else:
                        print("未识别气味，停止播放")
                        stop_cmd = stop_play()
                        await client.write_gatt_char(SCENT_WRITE_UUID, stop_cmd)
                        last_recognized_smell = None  # 如果未识别则重置记录状态

                # 每秒更新一次
                await asyncio.sleep(1)

            print("ScentDevice 控制已停止")

    except Exception as e:
        print(f"控制 ScentDevice 时发生错误: {e}")
    finally:
        scentdevice_is_running = False  # 恢复运行标志
        scent_client = None  # 清空蓝牙客户端
        print("已断开 ScentDevice 连接")




# 创建按钮
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

btn_connect = tk.Button(button_frame, text="连接蓝牙", command=lambda: run_async_task(connect_bluetooth()))
btn_connect.pack(side=tk.LEFT, padx=5, pady=5)

btn_disconnect = tk.Button(button_frame, text="断开蓝牙", command=lambda: run_async_task(disconnect_bluetooth()))
btn_disconnect.pack(side=tk.LEFT, padx=5, pady=5)

btn_start = tk.Button(button_frame, text="开始读数", command=lambda: run_async_task(start_reading()))
btn_start.pack(side=tk.LEFT, padx=5, pady=5)

btn_stop = tk.Button(button_frame, text="停止读数", command=lambda: run_async_task(stop_reading()))
btn_stop.pack(side=tk.LEFT, padx=5, pady=5)

btn_clear = tk.Button(button_frame, text="清空数据", command=clear_data)
btn_clear.pack(side=tk.LEFT, padx=5, pady=5)

btn_thumbnail = tk.Button(button_frame, text="缩略图", command=switch_to_thumbnail)
btn_thumbnail.pack(side=tk.LEFT, padx=5, pady=5)

btn_full = tk.Button(button_frame, text="完整绘图", command=switch_to_full_plot)
btn_full.pack(side=tk.LEFT, padx=5, pady=5)

btn_save_all = tk.Button(button_frame, text="保存数据", command=save_all_sensor_data)
btn_save_all.pack(side=tk.LEFT, padx=5, pady=5)

btn_train = tk.Button(button_frame, text="训练模式", command=lambda: run_async_task(start_training_mode()))
btn_train.pack(side=tk.LEFT, padx=5, pady=5)

btn_recognition = tk.Button(button_frame, text="识别模式", command=lambda: run_async_task(start_recognition_mode()))
btn_recognition.pack(side=tk.LEFT, padx=5, pady=5)

btn_control_scent = tk.Button(button_frame, text="控制小播", command=lambda: asyncio.create_task(control_scent_device()))
btn_control_scent.pack(side=tk.LEFT, padx=5, pady=5)




# 添加信息栏
add_info_bar(root)


async def plot_task():
    """独立的绘图任务"""
    global is_reading
    while True:
        if is_reading:
            update_plot()
        await asyncio.sleep(0.1)  # 每 0.1 秒更新一次


async def main():
    """运行 tkinter 主循环"""
    asyncio.create_task(plot_task())  # 启动绘图协程

    while True:
        root.update()
        await asyncio.sleep(0.05)


# 启动 asyncio 主循环
asyncio.run(main())