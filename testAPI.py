import requests
import base64

image = open('image.jpg', 'rb')
image_read = image.read()
image_64_encode = base64.encodebytes(image_read)
r = requests.post('http://0.0.0.0:5000/', json={"data": image_64_encode.decode('utf-8')})

print(r.json())
