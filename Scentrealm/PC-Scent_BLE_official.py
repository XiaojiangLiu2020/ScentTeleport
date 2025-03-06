
import asyncio
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
from binascii import unhexlify
from crcmod import crcmod

# 设备的Characteristic UUID
par_notification_characteristic = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
# 设备的Characteristic UUID（具备写属性Write）
par_write_characteristic = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
# 设备的MAC地址
par_device_addr = "08:D1:F9:12:D5:36"

FRAME_HEAD = 'F5'
FRAME_TAIL = '55'


def crc16Add(str_data):
    crc16 = crcmod.mkCrcFun(0x18005, rev=True, initCrc=0xFFFF, xorOut=0x0000)
    data = str_data.replace(" ", "")
    readcrcout = hex(crc16(unhexlify(data))).upper()
    str_list = list(readcrcout)
    if len(str_list) < 6:
        str_list.insert(2, '0'*(6-len(str_list)))  # 位数不足补0
    crc_data = "".join(str_list)
    return crc_data[2:4]+' '+crc_data[4:]


def ten2sixteen(num, length):
    """
    十进制转十六进制
    :param num: 十进制数字
    :param length: 字节长度
    :return:
    """
    data = str(hex(eval(str(num))))[2:]
    data_len = len(data)
    if data_len % 2 == 1:
        data = '0' + data
        data_len += 1

    sixteen_str = "00 " * (length - data_len//2) + data[0:2] + ' ' + data[2:]
    return sixteen_str.strip()


def cmd2bytearray(cmd_str: str):
    verify = crc16Add(cmd_str)
    cmd = FRAME_HEAD + ' ' + cmd_str + ' ' + verify + ' ' + FRAME_TAIL
    print(cmd)
    return bytearray.fromhex(cmd)


def device_capluse():
    """
    获取设备气路胶囊信息
    :return:
    """
    cmd_data = '00 00 00 01 0E 01 06 00 00'
    return cmd2bytearray(cmd_data)


def start_play(scent: int, playtime: int):
    play_cmd = '00 00 00 01 02 05'
    scent_channel = ten2sixteen(scent, 1)
    if playtime == 0:  # 一直播放
        playtime16 = 'FF FF FF FF'
    else:
        playtime16 = ten2sixteen(playtime, 4)
    cmd_data = play_cmd + ' ' + scent_channel + ' ' + playtime16
    return cmd2bytearray(cmd_data)


def stop_play():
    """
    停止播放
    :return:
    """
    stop_cmd = '00 00 00 01 00 01 00'
    return cmd2bytearray(stop_cmd)


def status_check():
    """
    检查工作状态
    :return:
    """
    status_cmd = '00 00 00 01 11 01 00 00 00'
    return cmd2bytearray(status_cmd)


async def scan_devices():
    """
    扫描蓝牙设备
    :return:
    """
    devices = await BleakScanner.discover()
    for d in devices:  # d为类，其属性有：d.name为设备名称，d.address为设备地址
        print(d)


# 监听回调函数，此处为打印消息
def notification_handler(characteristic: BleakGATTCharacteristic, data: bytearray):
    # print("rev data:", data)
    print("rev data bytes2hex:", ' '.join(['%02x' % b for b in data]))


async def main():
    print("starting scan...")
    # 基于MAC地址查找设备
    device = await BleakScanner.find_device_by_address(
        par_device_addr, cb=dict(use_bdaddr=False)
    )
    if device is None:
        print("could not find device with address '%s'" % par_device_addr)
        return

        # 事件定义
    disconnected_event = asyncio.Event()

    # 断开连接事件回调
    def disconnected_callback(client):
        print("Disconnected callback called!")
        disconnected_event.set()

    print("connecting to device...")
    async with BleakClient(device, disconnected_callback=disconnected_callback) as client:
        print("Connected")
        await client.start_notify(par_notification_characteristic, notification_handler)

        await client.write_gatt_char(par_write_characteristic, device_capluse())  # 获取设备气路胶囊信息
        await asyncio.sleep(2.0)

        scent = 1  # 播放气路数
        playtime = 100000  # 播放时长，单位ms
        await client.write_gatt_char(par_write_characteristic, start_play(scent, playtime))  # 发送开始播放指令

        await asyncio.sleep(10.0)
        await client.write_gatt_char(par_write_characteristic, stop_play())  # 发送停止播放指令
        scent = 2  # 播放气路数
        playtime = 100000  # 播放时长，单位ms
        await client.write_gatt_char(par_write_characteristic, start_play(scent, playtime))  # 发送开始播放指令

        await asyncio.sleep(10.0)
        await client.write_gatt_char(par_write_characteristic, stop_play())
        scent = 3  # 播放气路数
        playtime = 100000  # 播放时长，单位ms
        await client.write_gatt_char(par_write_characteristic, start_play(scent, playtime))  # 发送开始播放指令

        await asyncio.sleep(10.0)
        await client.write_gatt_char(par_write_characteristic, stop_play())
        scent = 4  # 播放气路数
        playtime = 100000  # 播放时长，单位ms
        await client.write_gatt_char(par_write_characteristic, start_play(scent, playtime))  # 发送开始播放指令

        await asyncio.sleep(10.0)
        await client.write_gatt_char(par_write_characteristic, stop_play())

        await asyncio.sleep(10.0)
        await client.write_gatt_char(par_write_characteristic, status_check())  # 检查设备工作状态

        await client.stop_notify(par_notification_characteristic)
        await client.disconnect()


asyncio.run(main())
