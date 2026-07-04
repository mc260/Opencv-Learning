#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/*int main() {
    //image IO

    Mat image = imread("/home/ximeng/桌面/Week 3/1.jpg");
    //通道分离
    vector<Mat> channels;
    split(image, channels);
    Mat blueChannel = channels[0];   // B通道
    Mat greenChannel = channels[1];  // G通道
    Mat redChannel = channels[2];    // R通道

    // 通道运算 - 创建单通道图像和合成图像
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

    //转换为灰度图
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    //二值化
    Mat binary;
    threshold(gray, binary, 127, 255, THRESH_BINARY);

    //腐蚀和膨胀操作
    Mat eroded, dilated;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

    // 腐蚀：使前景对象变小
    erode(binary, eroded, kernel);

    // 膨胀：使前景对象变大
    dilate(binary, dilated, kernel);


    // 查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // 创建轮廓绘制图像
    Mat contourImage = image.clone();
    drawContours(contourImage, contours, -1, Scalar(0, 255, 0), 2);

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
    // 垂直拼接所有行
    vconcat(row1, row2, finalDisplay);
    vconcat(finalDisplay, row3, finalDisplay);
    vconcat(finalDisplay, row4, finalDisplay);

    // 调整最终显示大小
    resize(finalDisplay, finalDisplay, Size(1200, 1600));

    // 显示结果
    namedWindow("OpenCV 图像处理演示", WINDOW_NORMAL);
    imshow("OpenCV 图像处理演示", finalDisplay);
    waitKey(0);
}
*/