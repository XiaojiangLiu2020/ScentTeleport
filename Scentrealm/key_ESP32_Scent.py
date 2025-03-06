import asyncio
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
from binascii import unhexlify
from crcmod import crcmod

# ESP32 的 UUID
ESP32_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
ESP32_NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# ScentDevice 的 UUID 和 MAC 地址
SCENT_NOTIFICATION_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
SCENT_WRITE_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
SCENT_MAC_ADDR = "08:D1:F9:12:D5:36"

FRAME_HEAD = 'F5'
FRAME_TAIL = '55'


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


async def main():
    # 扫描 ESP32
    print("正在扫描 ESP32...")
    esp32_device = await BleakScanner.find_device_by_name("ESP32BLE")
    if not esp32_device:
        print("未找到 ESP32 设备")
        return

    # 扫描 ScentDevice
    print("正在扫描 ScentDevice...")
    scent_device = await BleakScanner.find_device_by_address(SCENT_MAC_ADDR)
    if not scent_device:
        print("未找到 ScentDevice")
        return

    async with BleakClient(esp32_device) as esp32_client, BleakClient(scent_device) as scent_client:
        print("已连接到 ESP32 和 ScentDevice")

        # 订阅 ESP32 按键通知
        def esp32_notification_handler(sender, data):
            key = data.decode()
            print(f"接收到 ESP32 按键: {key}")
            if key == "0":  # 按下 0 时停止播放
                print("停止气味播放")
                asyncio.create_task(scent_client.write_gatt_char(SCENT_WRITE_UUID, stop_play()))
            elif key.isdigit():  # 如果按键是数字，发送到 ScentDevice
                scent = int(key)
                print(f"播放气味通道: {scent}")
                play_cmd = start_play(scent, 10000)  # 10 秒播放
                asyncio.create_task(scent_client.write_gatt_char(SCENT_WRITE_UUID, play_cmd))

        await esp32_client.start_notify(ESP32_NOTIFY_UUID, esp32_notification_handler)

        # 等待输入或运行指定时间
        print("等待按键输入，运行中...")
        await asyncio.sleep(180)  # 运行 60 秒

        await esp32_client.stop_notify(ESP32_NOTIFY_UUID)

asyncio.run(main())