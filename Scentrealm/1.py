import asyncio
import json
import numpy as np
import matplotlib.pyplot as plt
from bleak import BleakClient, BleakScanner
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import os

# ESP32 的 UUID
ESP32_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
ESP32_NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
ESP32_WRITE_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

# 创建独立窗口
root = tk.Tk()
root.title("传感器阵列实时数据")
root.geometry("1200x1000")

# 全局变量初始化
sensor_data_lists = [[[] for _ in range(6)] for _ in range(6)]
time_points = []
last_update_time = time.time()
esp32_client = None  # 蓝牙客户端
is_reading = False  # 标志是否正在读数
data_queue = asyncio.Queue()  # 数据处理队列

# 图形和画布的全局变量
fig = None
axes = None
lines = None
canvas = None


def create_canvas():
    """创建画布和初始化子图"""
    global fig, axes, lines, canvas

    if canvas is not None:
        return  # 如果画布已经创建，则直接返回

    # 创建图形和子图
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

    # 将 matplotlib 图形嵌入 tkinter 窗口
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    print("画布已创建")


def update_plot():
    """更新实时数据的绘图"""
    global last_update_time, is_reading
    if not is_reading:  # 如果已经停止读数，则不更新绘图
        return

    current_time = time.time()
    if current_time - last_update_time < 0.1:  # 每 0.1 秒更新一次
        return

    last_update_time = current_time

    try:
        for i in range(6):
            for j in range(6):
                y_data = sensor_data_lists[i][j]
                if len(y_data) > 100:
                    y_data = y_data[-100:]
                x_data = list(range(len(y_data)))

                lines[i][j].set_data(x_data, y_data)

                if y_data:
                    axes[i][j].set_ylim(min(y_data) - 1000, max(y_data) + 1000)
                    axes[i][j].set_xlim(0, len(y_data))

        canvas.draw()  # 更新画布
    except Exception as e:
        print(f"绘图更新错误: {e}")



received_data = ""  # 全局变量，用于拼接数据

def esp32_notification_handler(sender, data):
    """处理 ESP32 的蓝牙通知"""
    global received_data, data_queue
    try:
        chunk = data.decode("utf-8").strip()
        received_data += chunk  # 拼接数据
        #rint(f"接收到的 chunk 数据: {chunk}")
        #print(f"当前拼接后的 received_data: {received_data}")

        if "#END#" in received_data:  # 如果接收到完整数据
            complete_data, received_data = received_data.split("#END#", 1)  # 分割完整数据
            asyncio.create_task(data_queue.put(complete_data))  # 将完整数据放入队列
    except Exception as e:
        print(f"通知处理错误: {e}")



async def process_data():
    """从队列中读取数据并处理"""
    global sensor_data_lists, is_reading
    while True:
        chunk = await data_queue.get()
        if not is_reading:
            continue  # 如果停止读数，则跳过数据处理

        try:
            if not chunk.strip():  # 检查是否为空字符串
                print("接收到空数据，跳过处理")
                continue

            #print(f"接收到的数据: {chunk}")  # 打印接收到的数据


            complete_data = chunk.strip()
            print(f"接收到完整数据: {complete_data}")

            try:
                sensor_matrix = json.loads(complete_data)
            except json.JSONDecodeError as e:
                print(f"解析数据时出错: {e}. 数据内容: {complete_data}")
                continue

            for i in range(6):
                for j in range(6):
                    sensor_data_lists[i][j].append(sensor_matrix[i][j])

            print(sensor_data_lists)

            # 更新绘图
            update_plot()

        except Exception as e:
            print(f"数据处理错误: {e}")

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

    create_canvas()  # 动态创建画布和子图

    await esp32_client.start_notify(ESP32_NOTIFY_UUID, esp32_notification_handler)
    print("已订阅通知")
    await esp32_client.write_gatt_char(ESP32_WRITE_UUID, "read_sensor_array".encode())
    print("已发送 read_sensor_array 指令")
    is_reading = True


async def stop_reading():
    """停止读数"""
    global esp32_client, is_reading
    is_reading = False  # 立即停止绘图更新

    if not esp32_client or not esp32_client.is_connected:
        print("蓝牙未连接")
        return

    try:
        await esp32_client.stop_notify(ESP32_NOTIFY_UUID)
        print("已停止通知")
        await esp32_client.write_gatt_char(ESP32_WRITE_UUID, "stop_sensor_array".encode())
        print("已发送 stop_sensor_array 指令")
    except Exception as e:
        print(f"停止读数时发生错误: {e}")


def clear_data():
    """清空传感器数据"""
    global sensor_data_lists, time_points
    sensor_data_lists = [[[] for _ in range(6)] for _ in range(6)]
    time_points = []
    print("已清空数据")
    if canvas:
        update_plot()


def run_async_task(coroutine):
    """将异步协程包装为任务"""
    asyncio.create_task(coroutine)


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


async def main():
    """运行 tkinter 主循环"""
    asyncio.create_task(process_data())  # 启动数据处理协程
    while True:
        root.update()
        await asyncio.sleep(0.1)  # 减小睡眠时间，提高响应速度


# 启动 asyncio 主循环
asyncio.run(main())