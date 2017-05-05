import socket
import csv
import fetch_girl_images as fgi

HOST, PORT = '', 1234

listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
girl_mark_dic = {}

print 'Serving HTTP on port %s ...' % PORT
while True:
    client_connection, client_address = listen_socket.accept()
    request = client_connection.recv(1024)
    request_line = request.splitlines()[0]
    request_line = request_line.rstrip('\r\n')
    # Break down the request line into components
    (request_method,
     path,
     request_version
     ) = request_line.split()
    (image, mark) = path.lstrip('/').split('?')
    print(image)
    print(mark)
    girl_mark_dic[image] = mark
    csvFile = open(fgi.GIRL_MARK_FILE, "w")
    writer = csv.writer(csvFile)
    for k, v in girl_mark_dic.iteritems():
        writer.writerow([k, v])
    csvFile.close()

    http_response = """
HTTP/1.1 200 OK

Hello, World!
"""
    client_connection.sendall(http_response)
    client_connection.close()