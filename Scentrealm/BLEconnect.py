import asyncio
from bleak import BleakClient, BleakScanner

# 替换为设备的蓝牙名称或 MAC 地址
DEVICE_NAME = "scent08d1f912d536"
CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # 替换为文档中的特性 UUID

# 相关蓝牙协议 https://github.com/Scentrealm/Bluetooth/blob/main/Bluetooth.md
# 生成播放单路气味的命令
def generate_command(channel, duration):
    """
    生成播放单路气味的蓝牙命令
    :param channel: 通道号 (1-10)
    :param duration: 持续时间 (1-255 秒)
    :return: 命令字节数组
    """
    if not (1 <= channel <= 10):
        raise ValueError("通道号必须在 1 到 10 之间")
    if not (1 <= duration <= 255):
        raise ValueError("持续时间必须在 1 到 255 秒之间")

    # 固定帧头
    command = [0xF5]

    # 源地址（高位和低位）
    source_address = [0x83, 0x03]
    command.extend(source_address)

    # 目标地址（高位和低位）
    target_address = [0x00, 0x03]
    command.extend(target_address)

    # 指令码（播放单路气味指令）
    command.append(0x02)

    # 数据长度（数据段长度，固定为 5 字节）
    command.append(0x05)

    # 数据段（气味编号和持续时间）
    # 气味编号（通道号）：1 字节
    # 持续时间（4 字节，毫秒）
    duration_ms = duration * 1000  # 将秒转换为毫秒
    data = [channel, (duration_ms >> 24) & 0xFF, (duration_ms >> 16) & 0xFF, (duration_ms >> 8) & 0xFF, duration_ms & 0xFF]
    command.extend(data)

    # 计算校验和（从源地址到数据段的所有字段的和，取低 2 字节）
    checksum = sum(command[1:]) & 0xFFFF
    command.extend([(checksum >> 8) & 0xFF, checksum & 0xFF])

    # 固定帧尾
    command.append(0x55)

    print(f"Generated command: {command}")
    return bytearray(command)


async def main():
    # 搜索设备
    print("Scanning for devices...")
    devices = await BleakScanner.discover()
    target_device = None
    for device in devices:
        print(f"Found device: {device.name} - {device.address}")
        if device.name == DEVICE_NAME:
            target_device = device
            break

    if not target_device:
        print(f"Device '{DEVICE_NAME}' not found.")
        return

    # 连接到设备
    print(f"Connecting to {DEVICE_NAME}...")
    async with BleakClient(target_device.address) as client:
        print(f"Connected to {DEVICE_NAME}")

        # 播放单路气味（示例：通道 1，持续 10 秒）
        channel = 1  # 替换为你想要播放的通道号
        duration = 10  # 替换为你想要的持续时间（秒）
        command = generate_command(channel, duration)

        print(f"Sending command to play channel {channel} for {duration} seconds...")
        await client.write_gatt_char(CHARACTERISTIC_UUID, command)
        print("Command sent successfully.")


# 运行主程序
asyncio.run(main())