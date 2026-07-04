#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 1. 读取图像
    Mat image = imread("/home/ximeng/桌面/Week 3/1.jpg"); // 请替换为你的图片路径
    if (image.empty()) {
        cout << "无法加载图像！请检查文件路径。" << endl;
        return -1;
    }
    
    // 调整图像大小以便显示
    resize(image, image, Size(600, 400));
    
    // 2. 转换为灰度图
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    
    // 3. 二值化
    Mat binary;
    threshold(gray, binary, 127, 255, THRESH_BINARY);
    
    // 4. 腐蚀和膨胀操作
    Mat eroded, dilated;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    
    // 腐蚀：使前景对象变小
    erode(binary, eroded, kernel);
    
    // 膨胀：使前景对象变大
    dilate(binary, dilated, kernel);
    
    // 5. 查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    
    // 在二值图像上查找轮廓
    findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    // 创建轮廓绘制图像
    Mat contourImage = image.clone();
    drawContours(contourImage, contours, -1, Scalar(0, 255, 0), 2);
    
    // 6. 通道分离
    vector<Mat> channels;
    split(image, channels);
    Mat blueChannel = channels[0];   // B通道
    Mat greenChannel = channels[1];  // G通道  
    Mat redChannel = channels[2];    // R通道
    
    // 7. 通道运算 - 创建单通道图像和合成图像
    Mat zeros = Mat::zeros(image.size(), CV_8UC1);
    
    // 只显示蓝色通道
    vector<Mat> blueChannels;
    blueChannels.push_back(blueChannel);
    blueChannels.push_back(zeros);
    blueChannels.push_back(zeros);
    Mat blueOnly;
    merge(blueChannels, blueOnly);
    
    // 只显示绿色通道
    vector<Mat> greenChannels;
    greenChannels.push_back(zeros);
    greenChannels.push_back(greenChannel);
    greenChannels.push_back(zeros);
    Mat greenOnly;
    merge(greenChannels, greenOnly);
    
    // 只显示红色通道
    vector<Mat> redChannels;
    redChannels.push_back(zeros);
    redChannels.push_back(zeros);
    redChannels.push_back(redChannel);
    Mat redOnly;
    merge(redChannels, redOnly);
    
    // 8. 通道运算 - 计算亮度 (使用公式: Y = 0.299*R + 0.587*G + 0.114*B)
    Mat weightedGray;
    Mat floatBlue, floatGreen, floatRed;
    
    // 转换为浮点数以便进行精确计算
    blueChannel.convertTo(floatBlue, CV_32F);
    greenChannel.convertTo(floatGreen, CV_32F);
    redChannel.convertTo(floatRed, CV_32F);
    
    // 应用权重并合并
    weightedGray = 0.114 * floatBlue + 0.587 * floatGreen + 0.299 * floatRed;
    weightedGray.convertTo(weightedGray, CV_8U);
    
    // 9. 显示所有结果
    // 创建一行显示原图、灰度图、二值图
    Mat row1, row2, row3, row4, finalDisplay;
    
    // 第一行：原图、灰度图、二值图
    hconcat(image, gray, row1);
    hconcat(row1, binary, row1);
    
    // 第二行：腐蚀、膨胀、轮廓
    hconcat(eroded, dilated, row2);
    hconcat(row2, contourImage, row2);
    
    // 第三行：蓝色通道、绿色通道、红色通道
    hconcat(blueOnly, greenOnly, row3);
    hconcat(row3, redOnly, row3);
    
    // 第四行：加权灰度图和其他信息
    Mat info = Mat::zeros(Size(600, 400), CV_8UC3);
    putText(info, "OpenCV 图像处理演示", Point(50, 100), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 255), 3);
    putText(info, "找到轮廓数量: " + to_string(contours.size()), Point(50, 180), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(info, "图像尺寸: " + to_string(image.cols) + "x" + to_string(image.rows), Point(50, 230), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    
    hconcat(weightedGray, info, row4);
    
    // 垂直拼接所有行
    vconcat(row1, row2, finalDisplay);
    vconcat(finalDisplay, row3, finalDisplay);
    vconcat(finalDisplay, row4, finalDisplay);
    
    // 调整最终显示大小
    resize(finalDisplay, finalDisplay, Size(1200, 1600));
    
    // 显示结果
    namedWindow("OpenCV 图像处理演示", WINDOW_NORMAL);
    imshow("OpenCV 图像处理演示", finalDisplay);
    
    // 10. 保存处理结果
    imwrite("gray_image.jpg", gray);
    imwrite("binary_image.jpg", binary);
    imwrite("contour_image.jpg", contourImage);
    imwrite("blue_channel.jpg", blueOnly);
    imwrite("green_channel.jpg", greenOnly);
    imwrite("red_channel.jpg", redOnly);
    
    cout << "处理完成！按任意键退出..." << endl;
    waitKey(0);
    
    return 0;
}