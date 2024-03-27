// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/log.h>
#include <iostream>
#include <jni.h>
#include <string>
#include <vector>
#include <array>
#include <platform.h>
#include <benchmark.h>
#include <cmath>
#include "yolo.h"
#include "ndkcamera.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON


static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

// Function to find the largest contour in a grayscale image
std::vector<cv::Point> findLargestContour(const cv::Mat& grayImage) {
    // Perform edge detection
    cv::Mat edges;
    cv::Canny(grayImage, edges, 100, 200);

    // Use morphological operations to close gaps between edge fragments
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat closed;
    cv::morphologyEx(edges, closed, cv::MORPH_CLOSE, kernel);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find the largest contour based on area
    std::vector<cv::Point> largestContour;
    double maxArea = 0.0;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            largestContour = contour;
        }
    }

    return largestContour;
}

cv::RotatedRect fitEllipse(const std::vector<cv::Point>& contour) {
    return cv::fitEllipse(contour);
}
// Function to calculate the foci of an ellipse
std::pair<cv::Point2f, cv::Point2f> calculateFoci(const cv::RotatedRect& ellipse) {
    // Extract the parameters of the ellipse
    cv::Point2f center = ellipse.center;
    float width = ellipse.size.width;
    float height = ellipse.size.height;
    float angle = ellipse.angle;

    // Calculate semi-major and semi-minor axes
    float a = std::max(width, height) / 2;
    float b = std::min(width, height) / 2;

    // Eccentricity calculation to find the distance from center to foci
    float c = std::sqrt(a*a - b*b);

    // Angle needs to be adjusted based on the orientation of the ellipse
    float angle_rad = cv::fastAtan2(b, a) * (CV_PI / 180.0f);
    float sin_angle = std::sin(angle_rad);
    float cos_angle = std::cos(angle_rad);

    // Calculate foci points
    cv::Point2f f1(center.x + c * cos_angle, center.y + c * sin_angle);
    cv::Point2f f2(center.x - c * cos_angle, center.y - c * sin_angle);

    return {f1, f2};
}
//GLCM
cv::Mat calculateGLCM(const cv::Mat& roi, int levels = 256) {
    cv::Mat glcm = cv::Mat::zeros(cv::Size(levels, levels), CV_32F);
    for (int y = 0; y < roi.rows; ++y) {
        for (int x = 0; x < roi.cols - 1; ++x) {
            int i = roi.at<uchar>(y, x);
            int j = roi.at<uchar>(y, x + 1);
            glcm.at<float>(i, j) += 1.0;
        }
    }
    return glcm;
}
//cc
double calculateContrast(const cv::Mat& glcm) {
    double contrast = 0.0;
    for (int i = 0; i < glcm.rows; ++i) {
        for (int j = 0; j < glcm.cols; ++j) {
            contrast += pow(i - j, 2) * glcm.at<float>(i, j);
        }
    }
    return contrast;
}
std::vector<double> analyzeTextureAroundFoci(const cv::Mat& image, const std::vector<cv::Point>& foci, int radius = 5) {
    std::vector<double> textureFeatures;
    for (const cv::Point& focus : foci) {
        int x1 = std::max(focus.x - radius, 0);
        int y1 = std::max(focus.y - radius, 0);
        int x2 = std::min(focus.x + radius, image.cols);
        int y2 = std::min(focus.y + radius, image.rows);
        cv::Mat roi = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        cv::Mat glcm = calculateGLCM(roi);
        double contrast = calculateContrast(glcm);
        textureFeatures.push_back(contrast);
    }
    return textureFeatures;
}

double calculateBearingAngle(const std::array<double, 2>& f1, const std::array<double, 2>& f2, double texture_f1, double texture_f2) {
    // 北方向向量，在图像坐标系中y轴方向为北
    std::array<double, 2> north_vector = {0, -1};

    // 根据纹理分析确定的入射方向
    std::array<double, 2> incident_vector;
    if (texture_f1 > texture_f2) {
        // 入射方向从f2到f1
        incident_vector = {f1[0] - f2[0], f1[1] - f2[1]};
    } else {
        // 入射方向从f1到f2
        incident_vector = {f2[0] - f1[0], f2[1] - f1[1]};
    }

    // 归一化入射向量
    double norm = std::sqrt(incident_vector[0] * incident_vector[0] + incident_vector[1] * incident_vector[1]);
    std::array<double, 2> incident_vector_normalized = {incident_vector[0] / norm, incident_vector[1] / norm};

    // 计算入射向量与北方向量之间的角度
    double dot_product = incident_vector_normalized[0] * north_vector[0] + incident_vector_normalized[1] * north_vector[1];
    double angle_radians = std::acos(dot_product);
    double angle_degrees = angle_radians * (180.0 / M_PI);

    // 根据x坐标调整角度到正确的象限
    if (incident_vector[0] < 0) {
        angle_degrees = 360 - angle_degrees;
    }

    return angle_degrees;
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static Yolo* g_yolo = 0;
static ncnn::Mutex lock;

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // nanodet
    {
        ncnn::MutexLockGuard g(lock);

        if (g_yolo)
        {
            std::vector<Object> objects;
            g_yolo->detect(rgb, objects);

            g_yolo->draw(rgb, objects);
        }
        else
        {
            draw_unsupported(rgb);
        }
    }

    draw_fps(rgb);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_yolo;
        g_yolo = 0;
    }

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL
Java_com_camsnoar_glasscrush_Yolov8Ncnn_loadModel(JNIEnv *env, jobject thiz, jobject assetManager,
                                                  jint modelid, jint cpugpu) {
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1) {
        return JNI_FALSE;
    }

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char *modeltypes[] =
            {
                    "bestm",
//        "s",
            };

    const int target_sizes[] =
            {
//                    640,
                    320,
            };

    const float mean_vals[][3] =
            {
//        {103.53f, 116.28f, 123.675f},
                    {103.53f, 116.28f, 123.675f},
            };

    const float norm_vals[][3] =
            {
//        { 1 / 255.f, 1 / 255.f, 1 / 255.f },
                    {1 / 255.f, 1 / 255.f, 1 / 255.f},
            };

    const char *modeltype = modeltypes[(int) modelid];
    int target_size = target_sizes[(int) modelid];
    bool use_gpu = (int) cpugpu == 1;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0) {
            // no gpu
            delete g_yolo;
            g_yolo = 0;
        } else {
            if (!g_yolo)
                g_yolo = new Yolo;
            g_yolo->load(mgr, modeltype, target_size, mean_vals[(int) modelid],
                         norm_vals[(int) modelid], use_gpu);
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jstring JNICALL
Java_com_camsnoar_glasscrush_Yolov8Ncnn_detect(JNIEnv *env, jobject thiz, jobject bitmap) {

    //图片处理
    AndroidBitmapInfo info;
    void *pixels;
    AndroidBitmap_getInfo(env, bitmap, &info);
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    // 创建一个原始的 cv::Mat 对象，大小和类型与 Bitmap 匹配
    cv::Mat img(info.height, info.width, CV_8UC4, pixels);

    // 转换颜色空间从ARGB到RGB，同时移除Alpha通道
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGRA2RGB);

    // 你可能还需要调整图像尺寸以匹配你的YOLO模型输入尺寸
    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(320, 320));

    std::vector<Object> objects;
    g_yolo->detect(img_resized, objects);
    //需要知道他的mask掩码是什么样子的
    AndroidBitmap_unlockPixels(env, bitmap);
         std::string result;
        Object &detectedObject = objects[0];
        cv::Mat &mask = detectedObject.mask;// "CV_32F"
        cv::Mat mask_uint8;
        if (mask.depth() == CV_32F) {
        mask.convertTo(mask_uint8, CV_8UC1, 255.0); // 转换为8位整数并缩放
        } else {
        mask_uint8 = mask.clone(); // 如果已经是CV_8UC1，则直接复制
        }


        cv::Mat mask_bgr;
        cv::cvtColor(mask_uint8, mask_bgr, cv::COLOR_BGRA2BGR);
        // 对比度增强
        cv::Mat lab;
        cv::cvtColor(mask_bgr, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(lab_channels[0], lab_channels[0]);
        cv::merge(lab_channels, lab);
        cv::cvtColor(lab, mask_bgr, cv::COLOR_Lab2BGR);

        // 去噪
        cv::Mat denoised_mask;
        cv::bilateralFilter(mask_bgr, denoised_mask, 9, 75, 75);

        // 灰度转换
        cv::Mat preprocessed_gray;
        cv::cvtColor(denoised_mask, preprocessed_gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point> largestContour = findLargestContour(preprocessed_gray);//报错
        if (!largestContour.empty() && largestContour.size() >= 5) {
            cv::RotatedRect ellipse = fitEllipse(largestContour);
            auto [f1, f2] = calculateFoci(ellipse);
            float distance = std::hypot(f2.x - f1.x, f2.y - f1.y);

            // 封装f1和f2为vector
            std::vector<cv::Point> foci1 = {f1};
            std::vector<cv::Point> foci2 = {f2};

            std::vector<double> texture_features_f1 = analyzeTextureAroundFoci(preprocessed_gray,
                                                                               foci1, 10);
            std::vector<double> texture_features_f2 = analyzeTextureAroundFoci(preprocessed_gray,
                                                                               foci2, 10);

            // 确保至少有一个特征值
            double texture_f1 = texture_features_f1.empty() ? 0 : texture_features_f1[0];
            double texture_f2 = texture_features_f2.empty() ? 0 : texture_features_f2[0];

            std::string direction = (texture_f1 > texture_f2) ? "f2 to f1" : "f1 to f2";
            std::array<double, 2> af1 = {static_cast<double>(f1.x), static_cast<double>(f1.y)};
            std::array<double, 2> af2 = {static_cast<double>(f2.x), static_cast<double>(f2.y)};
            double bearing_angle = calculateBearingAngle(af1, af2, texture_f1, texture_f2);

           // result += "Direction: " + direction;
            result += ", Distance: " + std::to_string(distance) + "pixels";
            result += ", Bearing angle: " + std::to_string(static_cast<int>(bearing_angle)) +
                      " degrees";

    }
    // 使用 JNI 环境将 C++ 字符串转换成 Java 字符串并返回
    return env->NewStringUTF(result.c_str());

}

}


