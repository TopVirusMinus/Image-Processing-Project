import base64
import cv2
import math
import numpy as np
import random
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


def RobertCrossGradient(image, kernel_type):
    padded_image, start_row, end_row, start_col, end_col = pad_image(image,(2,2))
    Output_image = padded_image.copy()
    Robert_kernel1 = []
    
    if kernel_type == 0:
        Robert_kernel1 =[[0,-1],[1,0]]
    elif kernel_type == 1:
        Robert_kernel1 =[[-1,0],[0,1]]
        
    Result = 0
    for r in range(start_row,end_row):
        for c in range(start_col,end_col):
            rs = r;  
            Result = 0
            for i in range(2):
                cs = c
                for k in range(2):
                    Value = padded_image[rs,cs] * Robert_kernel1[i][k] 
                    Result += Value
                    cs = cs+ 1
                rs = rs + 1 

            Output_image[r][c] = Result  
    Output_image=unpad_image(Output_image, start_row, end_row, start_col, end_col)
    return Output_image

def GetBack_Values(kernel_size):
    Dic= {}
    Value = 3
    Diff = 2
    for i in range(0,50):
        Dic.update({Value :Value - Diff})
        Value += 2
        Diff += 1
    return Dic

def UnsharpAvgFilter(Kernel_size, image, K_Value=0.1):
    Kernel_size = (Kernel_size, Kernel_size)
    Padded_UnSharp_image,start_row2, end_row2, start_col2, end_col2 = pad_image(image,Kernel_size)
    Unsharp_Avg_image = Padded_UnSharp_image.copy()
    Original_image = Padded_UnSharp_image.copy()
    Mask_image = Padded_UnSharp_image.copy()
    
    Kernel_dimensions = Kernel_size[0] * Kernel_size[1]
    Back_Value = GetBack_Values(Kernel_size)
    V1 = 0
    Avg = 0
    Avg_Result = 0
    
    for r in range(start_row2,end_row2):
        for c in range(start_col2,end_col2):
            rstart = r - Back_Value[Kernel_size[0]]
            cstart = c - Back_Value[Kernel_size[0]]
            Avg = 0
            i = 0
            while i < Kernel_size[0] and rstart < end_row2 and rstart >= start_row2:
                k = 0
                while k < Kernel_size[0] and cstart < end_col2 and cstart >= start_col2:
                    V1 = Padded_UnSharp_image[rstart][cstart]
                    Avg += V1
                    cstart += 1
                    k +=1
                i += 1
                rstart += 1

            Avg_Result = Avg / Kernel_dimensions
            Unsharp_Avg_image[r][c] =  Avg_Result
    for r in range(start_row2,end_row2):
      for c in range(start_col2,end_col2):
        Mask_image[r][c] = Padded_UnSharp_image[r][c] - Unsharp_Avg_image[r][c]
        
    for r in range(start_row2,end_row2):
        for c in range(start_col2,end_col2):
            Padded_UnSharp_image[r][c] += (K_Value * Mask_image[r][c])
    
    Final_UnSharp_image= unpad_image(Padded_UnSharp_image,start_row2, end_row2, start_col2, end_col2)
    return Final_UnSharp_image

def HighboostFilter(Kernel_size, image, K_Value=1.1):
    Kernel_size = (Kernel_size, Kernel_size)
    Padded_UnSharp_image,start_row2, end_row2, start_col2, end_col2 = pad_image(image,Kernel_size)
    Unsharp_Avg_image = Padded_UnSharp_image.copy()
    Original_image = Padded_UnSharp_image.copy()
    Mask_image = Padded_UnSharp_image.copy()

    
    Kernel_dimensions = Kernel_size[0] * Kernel_size[1]
    Back_Value = GetBack_Values(Kernel_size)
    V1 = 0
    Avg = 0
    Avg_Result = 0
    
    for r in range(start_row2,end_row2):
        for c in range(start_col2,end_col2):
            rstart = r - Back_Value[Kernel_size[0]]
            cstart = c - Back_Value[Kernel_size[0]]
            Avg = 0
            i = 0
            while i < Kernel_size[0] and rstart < end_row2 and rstart >= start_row2:
                k = 0
                while k < Kernel_size[0] and cstart < end_col2 and cstart >= start_col2:
                    V1 = Padded_UnSharp_image[rstart][cstart]
                    Avg += V1
                    cstart += 1
                    k +=1
                i += 1
                rstart += 1

            Avg_Result = Avg / Kernel_dimensions
            Unsharp_Avg_image[r][c] =  Avg_Result
    for r in range(start_row2,end_row2):
      for c in range(start_col2,end_col2):
        Mask_image[r][c] = Padded_UnSharp_image[r][c] - Unsharp_Avg_image[r][c]

    for r in range(start_row2,end_row2):
        for c in range(start_col2,end_col2):
            Original_image[r][c] += (K_Value * Mask_image[r][c])
            
    Highboost_image= unpad_image(Original_image,start_row2, end_row2, start_col2, end_col2)
    return Highboost_image


def LaplaceOperator(Kernel_size, image, kernel_type=0):
    Kernel_size = (Kernel_size, Kernel_size)
    Laplace_paded_image, start_row3, end_row3, start_col3, end_col3 = pad_image(image,Kernel_size)
    Back_Value = GetBack_Values(Kernel_size)
    Laplace_image1=Laplace_paded_image.copy()
    Laplace_kernel1 =[]
    
    if kernel_type == 0:
        Laplace_kernel1 = [[0,1,0],[1,-4,1],[0,1,0]]
    elif kernel_type == 1:
        Laplace_kernel1 = [[0,-1,0],[-1,4,-1],[0,-1,0]]
    elif kernel_type == 2:
        Laplace_kernel1 = [[1,1,1],[1,-8,1],[1,1,1]]
    elif kernel_type == 3:
        Laplace_kernel1 = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    
    V1 = 0
    Result = 0
    for r in range(start_row3,end_row3):
        for c in range(start_col3,end_col3):
            rstart = r - Back_Value[Kernel_size[0]]
            cstart = c - Back_Value[Kernel_size[0]]
            Result = 0
            i = 0
            while i < Kernel_size[0] and rstart < end_row3 and rstart >= start_row3:
                k = 0
                while k < Kernel_size[0] and cstart < end_col3 and cstart >= start_col3:
                    V1 = Laplace_paded_image[rstart][cstart] * Laplace_kernel1[i][k]
                    Result += V1
                    cstart += 1
                    k +=1
                i += 1
                rstart += 1
            Laplace_image1[r][c] = Result

    Laplace_image1 = unpad_image(Laplace_image1,start_row3,end_row3,start_col3,end_col3)
    return Laplace_image1

def SobelOperator(Kernel_size, image, kernel_type=0):
    Kernel_size =(Kernel_size,Kernel_size)
    Sobel_paded_image, start_row4, end_row4, start_col4, end_col4 = pad_image(image,Kernel_size)
    Kernel_dimensions = Kernel_size[0] * Kernel_size[1]
    Back_Value = GetBack_Values(Kernel_size)
    
    Sobel_image1 = Sobel_paded_image.copy()
    Sobel_kernel1 = []
    if kernel_type == 0:
        Sobel_kernel1 = [[-1,-2,-1],[0,0,0],[1,2,1]]
    elif kernel_type == 1:
        Sobel_kernel1 = [[-1,0,-1],[-2,0,2],[-1,0,1]]
    
    V1 = 0
    Result = 0
    for r in range(start_row4,end_row4):
        for c in range(start_col4,end_col4):
            rstart = r - Back_Value[Kernel_size[0]]
            cstart = c - Back_Value[Kernel_size[0]]
            Result = 0
            i = 0
            while i < Kernel_size[0] and rstart < end_row4 and rstart >= start_row4:
                k = 0
                while k < Kernel_size[0] and cstart < end_col4 and cstart >= start_col4:
                    V1 = Sobel_paded_image[rstart][cstart] * Sobel_kernel1[i][k]
                    Result += V1
                    cstart += 1
                    k +=1
                i += 1
                rstart += 1
            Sobel_image1[r][c] = Result  
            
    Sobel_image1 = unpad_image(Sobel_image1,start_row4,end_row4,start_col4,end_col4)
    return Sobel_image1


def apply_impulse_noise(image):
    new_image = image.copy()
    for r in range(len(image)):
        for c in range(len(image[r])):
            new_image[r][c] = np.random.choice([0, 255, image[r][c]], p=(0.05, 0.05,0.90))
    return new_image

def apply_gaussian_noise(image, mean, stdev):
    gaussian_noise_image = image.copy()

    gaussian_noise_image = gaussian_noise_image.astype(float)

    for i in range(gaussian_noise_image.shape[0]):
        for j in range(gaussian_noise_image.shape[1]):
            u1 = np.random.rand()
            u2 = np.random.rand()

            # Box-Muller transform
            z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

            z1 = z1 * stdev + mean
            z2 = z2 * stdev + mean

            gaussian_noise_image[i, j] += z1
            if i+1 < gaussian_noise_image.shape[0]:
                gaussian_noise_image[i+1, j] += z2

    # Clip the values to be in the valid range 0-255
    gaussian_noise_image = np.clip(gaussian_noise_image, 0, 255)

    return gaussian_noise_image
@app.route('/process-image', methods=['POST'])
def process_image():
    image_data = request.form.get('image_data')
    filter_type = request.form.get('filter_type')
    kernel_size = int(request.form.get('kernel_size'))
    extra_parameters = request.form.get('extraParams').split(',')
    
    print(filter_type)
    print(extra_parameters)
    
    image_data = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filter_functions = {
        'MedianFilter': lambda: MedianFilter(kernel_size, gray_image),
        'AveragingFilter': lambda: AveragingFilter(kernel_size, gray_image),
        'MaxFilter': lambda: MaxFilter(kernel_size, gray_image),
        'MinimumFilter': lambda: MinimumFilter(kernel_size, gray_image),
        'RobertCrossGradient': lambda: RobertCrossGradient(gray_image, float(extra_parameters[0])),
        'UnsharpAvgFilter': lambda: UnsharpAvgFilter(kernel_size, gray_image, float(extra_parameters[0])),
        'HighboostFilter': lambda: HighboostFilter(kernel_size, gray_image, float(extra_parameters[0])),
        'LaplaceOperator': lambda: LaplaceOperator(kernel_size, gray_image, int(extra_parameters[0])),
        'SobelOperator': lambda: SobelOperator(kernel_size, gray_image, int(extra_parameters[0])),
        'apply_impulse_noise': lambda: apply_impulse_noise(gray_image),
        'apply_gaussian_noise': lambda: apply_gaussian_noise(gray_image, int(extra_parameters[0]), int(extra_parameters[1])),
    }
    
    processed_image = []
    if filter_type in filter_functions:
        processed_image = filter_functions[filter_type]()
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

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
