from flask import  request
def predict_result(image_path):
    # Initialize image path
    PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.
    return request.post(PyTorch_REST_API_URL, files=payload)
    
        
if __name__ == '__main__':
    predict_result('./data_image/test.jpg')