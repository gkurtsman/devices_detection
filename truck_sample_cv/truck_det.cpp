#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

// std::string protoFile = "/home/krtsmn/Develop/comp_vision/workspace/volume/src/truck/caffe_dnn/truck_features/model/mobilenet_ssd/mobilenet-ssd.prototxt";
std::string protoFile = "/home/krtsmn/Develop/comp_vision/dnn/devices_detection/model/caffe_modelzoo_model/deploy.prototxt";
// std::string protoFile = "/home/krtsmn/Develop/comp_vision/dnn/MobileNet-SSD/deploy.prototxt";

// std::string weightsFile = "/home/krtsmn/Develop/comp_vision/workspace/volume/src/truck/caffe_dnn/truck_features/model/mobilenet_ssd/mobilenet-ssd.caffemodel";
//std::string  weightsFile = "/home/krtsmn/Develop/comp_vision/dnn/devices_detection/model/caffe_modelzoo_model/ZF_faster_rcnn_final.caffemodel";
std::string weightsFile = "/home/krtsmn/Develop/comp_vision/dnn/MobileNet-SSD/mobilenet_iter_73000.caffemodel";

std::string imageFile = "/home/krtsmn/Develop/comp_vision/dnn/devices_detection/images/truck.jpg";
// std::string imageFile = "/home/krtsmn/Develop/comp_vision/dnn/devices_detection/images/car_audi.jpg";

std::string device = "cpu";

int main(int argc, char **argv)
{

	int inWidth = 300;
	int inHeight = 300;
	float thresh = 0.1;    

	cv::Mat frame = cv::imread(imageFile);
	cv::Mat frameCopy = frame.clone();
	int frameWidth = frame.cols;
	int frameHeight = frame.rows;
	printf ("frame width = %d\n",frameWidth);
	printf ("frame height = %d\n",frameHeight);

	cv::dnn::Net net = cv::dnn::readNetFromCaffe(protoFile, weightsFile);
	std::cout << "Using CPU device" << std::endl;
	net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);

	cv::Mat inpBlob = cv::dnn::blobFromImage(frame, 1.0 / 16, cv::Size(inWidth, inHeight),
		cv::Scalar(0, 0, 0), false, false);
	printf ("%s,%d: inpBlob.dims = %d\n",__FILE__,__LINE__, inpBlob.dims);
	net.setInput(inpBlob);
#if 1
	cv::Mat output = net.forward();
#else
	net.forward(cv::String("fc7"));
	cv::Mat output;
#endif
	int rows = output.size[2];
	int cols = output.size[3];
	printf ("output.channels=%d\n",output.size[1]);
	printf ("rows = %d,cols=%d\n",output.size[2],output.size[3]);
	printf ("&rows = %p\n",&rows);
	int actual_rows = (rows < 10) ? rows : 10;
	int actual_cols = (cols < 10) ? cols : 10;
	printf ("actual rows = %d\n", actual_rows);
	printf ("actual cols = %d\n", actual_cols);
	printf ("data type = %d\n", output.type());
	printf ("data size = %d\n", sizeof (output.data[0]));
	printf ("ptr size = %d\n", sizeof (*output.ptr(0,0)));
	printf ("sizeof CV32F = %d\n",sizeof(CV_32F));
	printf ("output matrix:\n");
	for (int i=0; i < actual_rows; ++i) {
		for (int j=0; j < actual_cols; ++j) {
			printf ("%g ",(float)(output.data[(i*actual_cols + j)*sizeof(float)]));
#if 0
			int ind = (i*actual_cols + j)*sizeof(CV_32F);
			for (int k=0; k < sizeof(CV_32F); ++k)
				printf ("%02x",((char*)output.data)[ind + k]);
			printf (" ");
#endif
		}
		printf ("\n");
	}
	float prob = (float)output.data[((0 * actual_cols + 2)) * sizeof(float)];
	printf ("prob=%g\n",prob);
#if 1
	float pic_coeff_width = frameWidth / inWidth;
	float pic_coeff_height = frameHeight / inHeight;
#else
	float pic_coeff_width = 1;
	float pic_coeff_height = 1;
#endif
	float xmin = ((float*)(output.data))[((0 * actual_cols + 3)) /* * sizeof(float)*/] * pic_coeff_width;
	float ymin = ((float*)(output.data))[((0 * actual_cols + 4)) /* * sizeof(float)]*/] * pic_coeff_height;
	float xmax = ((float*)(output.data))[((0 * actual_cols + 5)) /* * sizeof(float) */] * pic_coeff_width;
	float ymax = ((float*)(output.data))[((0 * actual_cols + 6)) /* * sizeof(float)*/] * pic_coeff_height;

//	cv::rectangle(frame,cv::Point(xmin,ymin), cv::Point(xmax,ymax), cv::Scalar(255,0,255),4);
	cv::rectangle(frame,cv::Point(xmin,ymin), cv::Point(xmax,ymax), cv::Scalar(0,255,0),4);
	cv::imshow ("image",frame);
	cv::waitKey();
	return 0;
}
