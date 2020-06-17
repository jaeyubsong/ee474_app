import requests
import json
import cv2

addr = 'http://localhost:7007'
test_url = addr + '/emotion'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('res/mask/surgical_mask.png')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)

payload = {"landmark": [1,2,3,4,5]}

img_file = {'file': ('image.jpg', img_encoded.tostring(), 'image/jpeg', {'Expires': '0'}),
            'json': (None, json.dumps(payload), 'application/json'),}
# send http request with image and receive response
response = requests.post(test_url, files=img_file)
# decode response
print(response.text)

# expected output: {u'message': u'image received. size=124x124'}