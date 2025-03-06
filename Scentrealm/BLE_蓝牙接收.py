import asyncio
import json
import numpy as np
import matplotlib.pyplot as plt
from bleak import BleakClient, BleakScanner
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# ESP32 的 UUID
ESP32_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
ESP32_NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
ESP32_WRITE_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # 用于向 ESP32 写入数据

# 创建独立窗口
root = tk.Tk()
root.title("传感器阵列实时数据")
root.geometry("1200x1000")

# 全局变量初始化
sensor_data_lists = [[[] for _ in range(6)] for _ in range(6)]
time_points = []
last_update_time = time.time()

# 创建图形和画布
fig, axes = plt.subplots(6, 6, figsize=(12, 12))
fig.tight_layout()
lines = [[None for _ in range(6)] for _ in range(6)]

# 将matplotlib图形嵌入tkinter窗口
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

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


def update_plot():
    global last_update_time
    current_time = time.time()

    # 限制更新频率
    if current_time - last_update_time < 0.1:  # 100ms 更新一次
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
        root.update()  # 更新tkinter窗口

    except Exception as e:
        print(f"绘图更新错误: {e}")


received_data = ""  # 初始化全局变量
def esp32_notification_handler(sender, data):
    global received_data, time_points, sensor_data_lists

    try:
        chunk = data.decode("utf-8").strip()
        received_data += chunk

        if "#END#" in received_data:
            complete_data, received_data = received_data.split("#END#", 1)
            print(f"接收到完整数据: {complete_data}")
            sensor_matrix = json.loads(complete_data)

            if len(time_points) == 0:
                time_points = list(range(1))
            else:
                time_points.append(time_points[-1] + 1)

            for i in range(6):
                for j in range(6):
                    sensor_data_lists[i][j].append(sensor_matrix[i][j])

            # 实时更新绘图
            update_plot()

    except json.JSONDecodeError as e:
        print(f"解析数据时出错: {e}")
    except Exception as e:
        print(f"其他错误: {e}")


async def main():
    print("正在扫描 ESP32...")
    esp32_device = await BleakScanner.find_device_by_name("ESP32BLE")
    if not esp32_device:
        print("未找到 ESP32 设备")
        return

    async with BleakClient(esp32_device) as esp32_client:
        if not esp32_client.is_connected:
            print("蓝牙连接失败")
            return
        print("已连接到 ESP32")

        await esp32_client.start_notify(ESP32_NOTIFY_UUID, esp32_notification_handler)
        print("已订阅通知")

        command = "read_sensor_array"
        print(f"向 ESP32 发送命令: {command}")
        await esp32_client.write_gatt_char(ESP32_WRITE_UUID, command.encode())

        try:
            while True:
                await asyncio.sleep(0.1)  # 减少睡眠时间提高响应性, 删除无法运行
                root.update()  # 确保窗口保持响应
        except KeyboardInterrupt:
            print("程序已停止")
        finally:
            await esp32_client.stop_notify(ESP32_NOTIFY_UUID)


# 运行主程序
asyncio.run(main())