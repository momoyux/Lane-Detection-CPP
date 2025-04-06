#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main() {
	// 读取输入视频（替换为你的视频路径）
	VideoCapture cap("/home/bb/project/lane_detection_project/road_video.mp4");
	if (!cap.isOpened()) {
		cerr << "Error opening video file!" << endl;
		return -1;
	}

	while (true) {
		Mat fframe;
		cap >> fframe; // 逐帧读取视频
		if (fframe.empty()) break;
		Mat frame;
    		cv::resize(fframe, frame, cv::Size(500,500));
		// ----------- 图像预处理 -----------
		// 1. 灰度化
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		// 2. 高斯模糊（核大小5x5）
		Mat blur_gray;
		GaussianBlur(gray, blur_gray, Size(5, 5), 0);

		// 3. Canny边缘检测（阈值50-150）
		Mat edges;
		Canny(blur_gray, edges, 50, 150);

		// ----------- ROI区域提取 -----------
		// 定义梯形ROI（根据视频分辨率调整坐标）
		Mat mask = Mat::zeros(edges.size(), CV_8UC1);
		Point roi_points[] = {
		Point(100, frame.rows), // 左下
		Point(100, frame.rows * 0.6), // 左上
		Point(400, frame.rows * 0.6), // 右上
		Point(frame.cols - 100, frame.rows) // 右下
		};
		fillConvexPoly(mask, roi_points, 4, Scalar(255));
		bitwise_and(edges, mask, edges); // 应用ROI

		// ----------- 霍夫变换检测直线 -----------
		vector<Vec4i> lines;
		HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 50, 10);

		// ----------- 筛选并绘制车道线 -----------
		vector<Vec4i> left_lines, right_lines;
		for (const auto& line : lines) {
			int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
			float slope = static_cast<float>(y2 - y1) / (x2 - x1 + 1e-5); // 避免除以零

			// 根据斜率筛选左右车道线
			if (abs(slope) < 0.5) continue; // 过滤水平线
			if (slope < 0) {
				left_lines.push_back(line); // 左车道线（负斜率）
			}
			else {
				right_lines.push_back(line); // 右车道线（正斜率）
			}
		}

		// ----------- 拟合左右车道线（平均法）-----------
		auto fit_line = [&](const vector<Vec4i>& lines, Scalar color) {
			if (lines.empty()) return;
			vector<Point> points;
			for (const auto& line : lines) {
				points.emplace_back(line[0], line[1]);
				points.emplace_back(line[2], line[3]);
			}
			Vec4f line_params;
			fitLine(points, line_params, DIST_L2, 0, 0.01, 0.01);

			// 根据拟合参数计算线段端点
			float vx = line_params[0], vy = line_params[1];
			float x0 = line_params[2], y0 = line_params[3];
			int y1 = frame.rows; // 线段底部（图像底部）
			int y2 = frame.rows / 2; // 线段顶部（图像中部）
			int x1 = static_cast<int>(x0 + (y1 - y0) * vx / vy);
			int x2 = static_cast<int>(x0 + (y2 - y0) * vx / vy);

			line(frame, Point(x1, y1), Point(x2, y2), color, 3, LINE_AA);
		};

		// 绘制左右车道线
		fit_line(left_lines, Scalar(0, 255, 0)); // 绿色左线
		fit_line(right_lines, Scalar(0, 0, 255)); // 红色右线

		// ----------- 显示结果 -----------
		imshow("Original Frame", frame);
		imshow("Canny Edges", edges);

		if (waitKey(25) == 27) break; // 按ESC退出
	}

	cap.release();
	destroyAllWindows();
	return 0;
}
