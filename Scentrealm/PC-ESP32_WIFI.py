import socket

# ESP32 的 IP 地址和端口号
esp32_ip = "192.168.1.100"
esp32_port = 8080

# 创建 TCP 客户端
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((esp32_ip, esp32_port))

# 发送数据到 ESP32
client.send(b"Hello ESP32!")

# 接收来自 ESP32 的响应
response = client.recv(1024)
print("Received from ESP32:", response.decode())

# 关闭连接
client.close()