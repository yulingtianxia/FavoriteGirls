import socket
import csv
import fetch_girl_images as fgi

HOST, PORT = '', 1234

listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)
girl_mark_dic = {}

csvfile = open(fgi.GIRL_MARK_FILE, "r")
reader = csv.reader(csvfile)
for item in reader:
    girl_mark_dic[item[0]] = item[1]

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

    girl_mark_dic[image] = mark
    print('No.' + str(len(girl_mark_dic)) + ' image:' + image + ' mark:' + str(mark))
    csvfile = open(fgi.GIRL_MARK_FILE, "w")
    writer = csv.writer(csvfile)
    for k, v in girl_mark_dic.iteritems():
        writer.writerow([k, v])
    csvfile.close()

    http_response = """
HTTP/1.1 200 OK

Hello, World!
"""
    client_connection.sendall(http_response)
    client_connection.close()