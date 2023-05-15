import base64
import cv2
import math
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def padding(image):
    img=np.zeros((image.shape[0]+2,image.shape[1]+2))
    for i in range(1,image.shape[0]+1):
        for k in range(1,image.shape[1]+1):
            img[i,k]=image[i-1][k-1]
    return img

def pad_image(image, kernel_size):
    height, width = image.shape
    pad_height = kernel_size[0] // 2
    pad_width = kernel_size[1] // 2
    padded_image = np.zeros((height + 2*pad_height, width + 2*pad_width))
    padded_image[pad_height:height+pad_height, pad_width:width+pad_width] = image
    start_row = pad_height
    end_row = start_row + height
    start_col = pad_width
    end_col = start_col + width
    return padded_image, start_row, end_row, start_col, end_col

def unpad_image(padded_image, start_row, end_row, start_col, end_col):
    unpadded_image = padded_image[start_row:end_row, start_col:end_col]
    return unpadded_image

def MedianFilter(Kernel_size, image):
    kernel=[]
    k=[Kernel_size,Kernel_size]
    r2=0
    c2=0
    img,start_row, end_row, start_col, end_col= pad_image(image,k)
    Image_After=np.empty((img.shape[0],img.shape[1]))

    while r2 <= img.shape[0]-k[0]:
        kernel.clear()
        for r in range(r2,r2+k[0]):
            for c in range(c2,c2+k[1]):
                #print(r,c)
                kernel.append(img[r,c])
                if np.isnan(Image_After[r,c]):
                    Image_After[r,c] = img[r,c]

        kernel.sort()
        Image_After[r2+math.floor(k[0]/2),c2+math.floor(k[1]/2)] = kernel[math.floor(len(kernel)/2)]

        if c2 == img.shape[1]-k[1]:
            c2=0
            r2+=1
        else:
            c2+=1


    Image_After=unpad_image(Image_After, start_row, end_row, start_col, end_col)            
    return Image_After

def AveragingFilter(Kernel_size, image):
    kernel=0
    k=[Kernel_size,Kernel_size]
    r2=0
    c2=0
    img,start_row, end_row, start_col, end_col= pad_image(image,k)
    Image_After=np.zeros((img.shape[0],img.shape[1]))

    while r2 <= img.shape[0]-k[0]:
        kernel=0
        for r in range(r2,r2+k[0]):
            for c in range(c2,c2+k[1]):
                kernel+=img[r,c]
    
        Image_After[r2+math.floor(k[0]/2),c2+math.floor(k[1]/2)] = kernel/(k[0]**2) 

        if c2 == img.shape[1]-k[1]:
            c2=0
            r2=r2+1
        else:
            c2=c2+1

    Image_After=unpad_image(Image_After, start_row, end_row, start_col, end_col)
    return Image_After

def MaxFilter(Kernel_size, image):
    kernel=[]
    k=[Kernel_size,Kernel_size]
    r2=0
    c2=0
    img,start_row, end_row, start_col, end_col= pad_image(image,k)
    Image_After=np.empty((img.shape[0],img.shape[1]))

    while r2 <= img.shape[0]-k[0]:
        kernel.clear()
        for r in range(r2,r2+k[0]):
            for c in range(c2,c2+k[1]):
                kernel.append(img[r,c])
                if np.isnan(Image_After[r,c]):
                    Image_After[r,c] = img[r,c]

        kernel.sort()
        Image_After[r2+math.floor(k[0]/2),c2+math.floor(k[1]/2)] = kernel[len(kernel)-1]

        if c2 == img.shape[1]-k[1]:
            c2=0
            r2+=1
        else:
            c2+=1


    Image_After=unpad_image(Image_After, start_row, end_row, start_col, end_col)
    return Image_After
    
def MinimumFilter(Kernel_size, image):
    kernel = []
    k = [Kernel_size, Kernel_size]
    r2 = 0
    c2 = 0
    img, start_row, end_row, start_col, end_col = pad_image(image, k)
    Image_After = np.empty(img.shape)

    while r2 <= img.shape[0] - k[0]:
        kernel.clear()
        for r in range(r2, r2 + k[0]):
            for c in range(c2, c2 + k[1]):
                kernel.append(img[r, c])
                if np.isnan(Image_After[r, c]):
                    Image_After[r, c] = img[r, c]

        kernel.sort()
        Image_After[r2 + k[0] // 2, c2 + k[1] // 2] = kernel[0]

        if c2 == img.shape[1] - k[1]:
            c2 = 0
            r2 += 1
        else:
            c2 += 1

    Image_After = unpad_image(Image_After, start_row, end_row, start_col, end_col)
    return Image_After


@app.route('/process-image', methods=['POST'])
def process_image():
    image_data = request.form.get('image_data')
    filter_type = request.form.get('filter_type')
    kernel_size = int(request.form.get('kernel_size'))
    
    print(filter_type)
    
    image_data = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if filter_type == 'MedianFilter':
        processed_image = MedianFilter(kernel_size, gray_image)
    elif filter_type == 'AveragingFilter':
        processed_image = AveragingFilter(kernel_size, gray_image)
    elif filter_type == 'MaxFilter':
        processed_image = MaxFilter(kernel_size, gray_image)
    elif filter_type == 'MinimumFilter':
        processed_image = MinimumFilter(kernel_size, gray_image)

    retval1, buffer1 = cv2.imencode('.jpg', processed_image)
    retval2, buffer2 = cv2.imencode('.jpg', gray_image)
    
    processed_image_base64 = base64.b64encode(buffer1).decode('utf-8')
    greyed_original_base64 = base64.b64encode(buffer2).decode('utf-8')
    print(np.count_nonzero(processed_image - gray_image))
    return {
        'original_greyed': f'data:image/jpeg;base64,{greyed_original_base64}',
        'processed_image': f'data:image/jpeg;base64,{processed_image_base64}'}

if __name__ == '__main__':
    app.run(debug=True)
