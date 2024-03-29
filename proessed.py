import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os
 
# Redefine the preprocess_image function according to the provided code
def preprocess_image(image):
    # Contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(11, 11))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Denoising
    denoised_img = cv2.bilateralFilter(enhanced_img, 9, 75, 75)

    # Grayscale conversion
    gray = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)

    return gray

def auto_canny(image, sigma=0.33):
    # 计算图像的中值
    v = np.median(image)
    # 根据中值自动确定阈值
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # 应用Canny边缘检测
    edged = cv2.Canny(image, lower, upper)
    return edged

def find_orientation(contour):
    # 将轮廓点转换为Nx2的数组
    data_pts = contour.reshape(-1, 2).astype(np.float32)
    
    # PCA分析
    mean, eigenvectors = cv2.PCACompute(data_pts, mean=None)
    
    # 主轴方向是第一个特征向量
    x_vector = eigenvectors[0][0]
    y_vector = eigenvectors[0][1]
    
    # 计算方向角度
    angle = np.arctan2(y_vector, x_vector)
    return angle
def find_largest_contour(gray_image):
    # 边缘检测
    edges =auto_canny(gray_image) 
    cv2.imshow("qwq",edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 闭运算连接轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Ellipse and Foci', closing)
    cv2.waitKey(0)
    # 轮廓检测
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    orientation = find_orientation(max_contour)

# 转换为度
    orientation_degree = np.degrees(orientation)
    print(orientation_degree)
    if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            ellipse = cv2.fitEllipse(largest_contour)
            # 绘制椭圆
            cv2.ellipse(gray_image, ellipse, (0, 255, 0), 2)
            cv2.imshow('Largest Ellipse', gray_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return largest_contour
    else:
        return None
    # 在原图上绘制椭圆
    for ellipse in filtered_contours:
        cv2.ellipse(image, ellipse, (0,255,0), 2)  # 绘制绿色椭圆
    cv2.imshow('Ellipse and Foci', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fit_ellipse(contour):
    # 在最大轮廓上拟合椭圆
    ellipse = cv2.fitEllipse(contour)
    return ellipse
# 计算焦点坐标，假设已经获得了椭圆的参数(ellipse)
def calculate_foci(ellipse):
    # Simplified for clarity and correctness
    (xc, yc), (d1, d2), angle = ellipse
    a = max(d1, d2) / 2
    b = min(d1, d2) / 2
    c = np.sqrt(a**2 - b**2)

    # 焦点相较于中心点的坐标
    foci_dx = c * np.sin(np.radians(angle))
    foci_dy = c * np.cos(np.radians(angle))

    f1x = int(xc + foci_dx)
    f1y = int(yc - foci_dy)
    f2x = int(xc - foci_dx)
    f2y = int(yc + foci_dy)

    return (f1x, f1y), (f2x, f2y)

 

def calculate_glcm(roi, levels=256):
    """
    计算给定区域（ROI）的灰度共生矩阵（GLCM）。
    :param roi: 区域的图像数据。
    :param levels: 图像的灰度级别数，默认为256。
    :return: GLCM矩阵。
    """
    glcm = np.zeros((levels, levels), dtype=int)
    for y in range(roi.shape[0]):
        for x in range(roi.shape[1] - 1):  # 防止索引越界
            i = roi[y, x]
            j = roi[y, x + 1]
            glcm[i, j] += 1
    return glcm

def calculate_contrast(glcm):
    """
    根据GLCM计算对比度。
    :param glcm: 灰度共生矩阵。
    :return: 对比度值。
    """
    contrast = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            contrast += (i - j) ** 2 * glcm[i, j]
    return contrast

# def analyze_texture_around_foci(image, foci, radius=5):
#     """
#     分析图像中指定焦点周围区域的纹理特征。
#     :param image: 图像数据，假设为灰度图。
#     :param foci: 焦点列表，每个焦点为(x, y)元组。
#     :param radius: 分析区域的半径。
#     :return: 各区域的对比度列表。
#     """
#     texture_features = []
#     for fx, fy in foci:
#         x1 = max(int(fx - radius), 0)
#         y1 = max(int(fy - radius), 0)
#         x2 = min(int(fx + radius), image.shape[1])
#         y2 = min(int(fy + radius), image.shape[0])
#         roi = image[y1:y2, x1:x2]
#         glcm = calculate_glcm(roi)
#         contrast = calculate_contrast(glcm)
#         texture_features.append(contrast)
#     return texture_features

# def analyze_texture_around_foci(image, foci, radius=5):
#     # 纹理分析在焦点周围的区域
#     texture_features = []

#     for fx, fy in foci:
#         x1 = max(int(fx - radius), 0)
#         y1 = max(int(fy - radius), 0)
#         x2 = min(int(fx + radius), image.shape[1])
#         y2 = min(int(fy + radius), image.shape[0])
#         roi = image[y1:y2, x1:x2]
#         glcm = graycomatrix(roi, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
#         contrast = graycoprops(glcm, 'contrast')[0, 0]
#         texture_features.append(contrast)
#     return texture_features

def analyze_texture_around_foci(image, foci, radius=5):
    texture_features = []
    for fx, fy in foci:
        x1 = max(int(fx - radius), 0)
        y1 = max(int(fy - radius), 0)
        x2 = min(int(fx + radius), image.shape[1])
        y2 = min(int(fy + radius), image.shape[0])
        # 确保裁剪区域有有效尺寸
        if x2 > x1 and y2 > y1:
            roi = image[y1:y2, x1:x2]
            # 检查ROI是否为空或大小为零
            if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                print("Warning: Empty ROI encountered. Skipping.")
                continue  # 或者你可以添加一个默认值：texture_features.append(default_value)
            glcm = calculate_glcm(roi)
            contrast = calculate_contrast(glcm)
            texture_features.append(contrast)
    return texture_features

def calculate_bearing_angle(f1, f2, texture_f1, texture_f2):
    # 以图像坐标系y轴方向为北
    north_vector = np.array([0, -1])

    # 根据纹理分析确定的入射方向构造入射向量
    if texture_f1 > texture_f2:
        # 入射方向是从 f2 到 f1
        incident_vector = np.array(f1) - np.array(f2)
    else:
        # 入射方向是从 f1 到 f2
        incident_vector = np.array(f2) - np.array(f1)

    # 计算入射向量和北方向量之间的角度
    incident_vector_normalized = incident_vector / np.linalg.norm(incident_vector)
    dot_product = np.dot(incident_vector_normalized, north_vector)
    angle_radians = np.arccos(dot_product)
    angle_degrees = np.degrees(angle_radians)

    # 根据x坐标调整角度到正确的象限
    if incident_vector[0] < 0:
        angle_degrees = 360 - angle_degrees

    return angle_degrees

# def calculate_bearing_angle(f1, f2, texture_f1, texture_f2):
#     """
#     修改为计算与图像x轴的角度，保持旋转不变性。
#     """
#     # x轴向量
#     x_axis_vector = np.array([1, 0])

#     # 计算两个焦点之间的向量
#     f_vector = np.array(f2) - np.array(f1)
    
#     # 确保方向性正确
#     if texture_f1 > texture_f2:
#         f_vector = -f_vector

#     # 计算向量与x轴的夹角
#     f_vector_normalized = f_vector / np.linalg.norm(f_vector)
#     dot_product = np.dot(f_vector_normalized, x_axis_vector)
#     angle_radians = np.arccos(dot_product)
#     angle_degrees = np.degrees(angle_radians)

#     # 调整角度到[0, 360)区间
#     if f_vector[1] < 0:
#         angle_degrees = 360 - angle_degrees

#     return angle_degrees
def calculate_bearing_angle(f1, f2, texture_f1, texture_f2):
    """
    修改为计算与图像x轴的角度，保持旋转不变性。
    """
    # x轴向量
    x_axis_vector = np.array([1, 0])

    # 计算两个焦点之间的向量
    f_vector = np.array(f2) - np.array(f1)
    
    # 确保方向性正确，如果纹理f1比纹理f2粗糙，意味着实际方向应该是从f2到f1
    if texture_f1 > texture_f2:
        f_vector = -f_vector

    # 计算向量与x轴的夹角
    f_vector_normalized = f_vector / np.linalg.norm(f_vector)
    dot_product = np.dot(f_vector_normalized, x_axis_vector)
    angle_radians = np.arccos(np.clip(dot_product, -1.0, 1.0))  # 避免超出[-1, 1]范围的数值导致错误
    angle_degrees = np.degrees(angle_radians)

    # 调整角度到[0, 360)区间
    if f_vector[1] < 0:
        angle_degrees = 360 - angle_degrees

    return angle_degrees

 

def find_ellipse_and_distance(largest_contour):
     ellipse = fit_ellipse (largest_contour)
     f1, f2 = calculate_foci (ellipse)
     distance = np.sqrt ((f2 [0] - f1 [0])**2 + (f2 [1] - f1 [1])**2)
     return f1, f2, distance
def processed(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Unable to load image at {image_path}")
        return None
    preprocessed_gray = preprocess_image(image)
    return preprocessed_gray

def analyze_image(preprocessed_gray,preprocessed_gray_croped):
    
    largest_contour = find_largest_contour(preprocessed_gray)
    if largest_contour is not None and len(largest_contour) >= 5:
        f1,f2,distance=find_ellipse_and_distance(largest_contour)
        texture_f1 = analyze_texture_around_foci(preprocessed_gray_croped, [f1], radius=10)
        texture_f2 = analyze_texture_around_foci(preprocessed_gray_croped, [f2], radius=10)
        if texture_f1 > texture_f2:
            direction = 'f2 to f1'
        else:
            direction = 'f1 to f2'
        bearing_angle = calculate_bearing_angle (f1, f2, texture_f1, texture_f2)
        return direction, bearing_angle, distance
    else:
        return None, None, None

# 主程序
image_path = '2024-01-17-08-51-17_cropped_1.jpg'
image = cv2.imread(image_path)

# 预处理图像
preprocessed_gray = preprocess_image(image)

# 使用预处理后的图像找到轮廓并拟合椭圆
largest_contour = find_largest_contour(preprocessed_gray)


if largest_contour is not None and len(largest_contour) >= 5:
    # 拟合椭圆和计算焦点
    ellipse = fit_ellipse(largest_contour)
    f1, f2 =calculate_foci(ellipse)
    distance = np.sqrt((f2[0] - f1[0])**2 + (f2[1] - f1[1])**2)
    # # 在原图上绘制椭圆和焦点
    cv2.ellipse(image, ellipse, (255, 0, 0), 2)
    cv2.line(image, (int(ellipse[0][0]), int(ellipse[0][1])), f1, (0, 255, 0), 2)
    cv2.line(image, (int(ellipse[0][0]), int(ellipse[0][1])), f2, (255, 0, 0), 2)
    # 使用之前定义的 analyze_texture_around_foci 函数来分析焦点周围的纹理
    texture_f1 = analyze_texture_around_foci(preprocessed_gray, [f1], radius=10)
    texture_f2 = analyze_texture_around_foci(preprocessed_gray, [f2], radius=10)
    # 判断纹理粗糙度，确定入射方向
    if texture_f1 > texture_f2:
        direction = 'f2 to f1'
    else:
        direction = 'f1 to f2'
    # 计算方位角
    bearing_angle = calculate_bearing_angle(f1, f2, texture_f1, texture_f2)
    print(direction)
    #两个焦点距离用来判断入射角
    print(distance)
    print("Bearing angle:", bearing_angle)

cv2.imshow('Ellipse and Foci', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# def find_image_pairs(folder_path, suffix1="_cropped_1.jpg", suffix2="_cropped_20.jpg"):
#     """
#     在给定文件夹中寻找匹配的图片对。
#     """
#     image_pairs = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(suffix1):
#             counterpart = filename.replace(suffix1, suffix2)
#             if counterpart in os.listdir(folder_path):
#                 image_pairs.append((filename, counterpart))
#     return image_pairs

# def analyze_image_pair(image_path1, image_path2):
#     preprocessed_gray1 = processed(image_path1)
#     preprocessed_gray2 = processed(image_path2)
#     if preprocessed_gray1 is None or preprocessed_gray2 is None:
#         print(f"Warning: Unable to process images {image_path1} or {image_path2}")
#         return None, None, None  # 或者定义一些默认值
#     direction, bearing_angle, distance = analyze_image(preprocessed_gray1, preprocessed_gray2)
#     return direction, bearing_angle, distance
    
#     return direction, bearing_angle, distance
# def process_and_analyze_folder(folder_path):
#     """
#     寻找并分析文件夹中的所有匹配的图片对。
#     """
#     results_file = open('analysis_results3.txt', 'w')
#     image_pairs = find_image_pairs(folder_path)
    
#     for image_path1, image_path2 in image_pairs:
#         full_path1 = os.path.join(folder_path, image_path1)
#         full_path2 = os.path.join(folder_path, image_path2)
#         direction, bearing_angle, distance = analyze_image_pair(full_path1, full_path2)
        
#         # 输出结果到文件
#         results_file.write(f"{image_path1}: Direction = {direction}, Bearing angle = {bearing_angle}, Distance = {distance}\n")
        
#     results_file.close()

# if __name__ == "__main__":
#     folder_path = 'outputs'
#     process_and_analyze_folder(folder_path)
if __name__=="__main__":
    pass
