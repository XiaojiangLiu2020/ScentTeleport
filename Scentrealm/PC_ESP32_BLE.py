import asyncio
from bleak import BleakClient, BleakScanner

SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_WRITE_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

async def main():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    esp32_address = None

    for device in devices:
        print(f"Found device: {device.name}, Address: {device.address}")
        if device.name == "ESP32BLE":
            esp32_address = device.address
            break

    if not esp32_address:
        print("ESP32 device not found. Make sure it is advertising.")
        return

    print(f"Connecting to ESP32 at {esp32_address}...")

    async with BleakClient(esp32_address) as client:
        print("Connected to ESP32!")

        def notification_handler(sender, data):
            try:
                print(f"Received notification from ESP32: {data.decode()}")
            except UnicodeDecodeError:
                print(f"Received raw data (unable to decode): {data}")

        print("Subscribing to notifications...")
        await client.start_notify(CHARACTERISTIC_NOTIFY_UUID, notification_handler)

        print("Subscribed to notifications!")
        await asyncio.sleep(300)  # 等待通知

        await client.stop_notify(CHARACTERISTIC_NOTIFY_UUID)
        print("Stopped notifications.")

asyncio.run(main())