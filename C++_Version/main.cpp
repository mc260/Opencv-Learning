#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

// 获取HSV颜色范围上下限
std::pair<cv::Scalar, cv::Scalar> get_limits(const cv::Scalar& color) {
    // 创建单像素图像用于颜色转换
    cv::Mat color_mat(1, 1, CV_8UC3, color);
    cv::Mat hsv_mat;
    cv::cvtColor(color_mat, hsv_mat, cv::COLOR_BGR2HSV);

    cv::Vec3b hsv_pixel = hsv_mat.at<cv::Vec3b>(0, 0);
    int hue = hsv_pixel[0];

    cv::Scalar lowerLimit, upperLimit;

    // 特殊处理：红色的问题
    if (hue >= 165) {
        lowerLimit = cv::Scalar(hue - 10, 100, 100);
        upperLimit = cv::Scalar(180, 255, 255);
    } else if (hue <= 15) {
        lowerLimit = cv::Scalar(0, 100, 100);
        upperLimit = cv::Scalar(hue + 10, 255, 255);
    } else {
        // 其他颜色处理
        lowerLimit = cv::Scalar(hue - 10, 100, 100);
        upperLimit = cv::Scalar(hue + 10, 255, 255);
    }

    return std::make_pair(lowerLimit, upperLimit);
}

// 获取掩码的边界框
cv::Rect get_bounding_box(const cv::Mat& mask) {
    // 如果掩码为空，返回空矩形
    if (mask.empty()) {
        return cv::Rect();
    }

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // 查找轮廓
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return cv::Rect();
    }

    // 合并所有轮廓的边界框
    cv::Rect bbox = cv::boundingRect(contours[0]);
    for (size_t i = 1; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        bbox |= rect; // 合并矩形
    }

    return bbox;
}

int main() {
    // 打开视频文件
    cv::VideoCapture video("/home/ximeng/CLionProjects/text/任务视频.mp4");

    if (!video.isOpened()) {
        std::cerr << "错误：无法打开视频文件" << std::endl;
        return -1;
    }

    // 红色 (BGR格式)
    cv::Scalar fingcolar(21,6,70);
    // 蓝色
    //cv::Scalar fingcolar(255, 255, 0);

    cv::Mat frame;

    while (true) {
        video >> frame;
        if (frame.empty()) {
            break;
        }

        // 转换为HSV颜色空间
        cv::Mat hsv_frame;
        cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);

        // 获取颜色范围并创建掩码
        auto limits = get_limits(fingcolar);
        cv::Mat mask;
        cv::inRange(hsv_frame, limits.first, limits.second, mask);

        // 进行形态学操作来减少噪声
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

        // 获取边界框
        cv::Rect bbox = get_bounding_box(mask);

        // 如果找到有效的边界框，绘制矩形
        if (bbox.width > 0 && bbox.height > 0) {
            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 255), 3); // 黄色边框，线宽5
        }

        // 显示帧
        cv::imshow("frame", frame);
        cv::imshow("mask", mask);

        // 按'q'退出
        if (cv::waitKey(40) == 'q') {
            break;
        }
    }

    video.release();
    cv::destroyAllWindows();

    return 0;
}