// Source: /home/krtsmn/Develop/comp_vision/dnn/caffe_ssd/examples/ssd/ssd_detect.cpp

#include <caffe/caffe.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cam.h"

_config::_config ()
{}

_config::_config (char *filename)
{
	std::string file;
	std::string file1;
	std::ifstream infile(filename);
	usemean = false;
	brightness = false;
	color = false;
	crop = false;
	while (infile >> file) {
		dbg ("file = %s", file.c_str());
		if (file.compare("USE_MEAN") == 0) {
			if (infile >> file1) {
				int d = std::stoi(file1);
				if (d == 1)
					usemean = true;
				else
					usemean = false;
			} else
				break;
		}
		if (file.compare("BRIGHTNESS") == 0) {
			if (infile >> file) {
				int d = std::stoi(file);
				if (d == 1)
					brightness = true;
				else
					brightness = false;
			} else
				break;
		}
		if (file.compare("COLOR") == 0) {
			if (infile >> file) {
				int d = std::stoi(file);
				if (d == 1)
					color = true;
				else
					color = false;
			} else
				break;
		}
		if (file.compare("CROP") == 0) {
			if (infile >> file) {
				int d = std::stoi(file);
				if (d == 1)
					crop = true;
				else
					crop = false;
			} else
				break;
		}
	}
}

class Detector {
public:
	Detector(const std::string& model_file,
		const std::string& weights_file,
		const std::string& mean_file,
		const std::string& mean_value,
		const _config config);

		std::vector<std::vector<float> > Detect(const cv::Mat& img);
		struct _config cfg;
private:

	void SetMean(const std::string& mean_file, const std::string& mean_value);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

	caffe::shared_ptr<caffe::Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
};

Detector::Detector(const std::string& model_file,
                const std::string& weights_file,
                const std::string& mean_file,
                const std::string& mean_value,
				const _config config) {
#ifdef CPU_ONLY
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

	cfg = config;
	/* Load the network. */
	net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
	net_->CopyTrainedLayersFrom(weights_file);
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
	if (cfg.usemean)
		SetMean(mean_file, mean_value);
}

std::vector<std::vector<float> > Detector::Detect(const cv::Mat& img) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
	dbg ("geopmetry=%d/%d",input_geometry_.height, input_geometry_.width);
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
	dbg ("input_channels=%d",input_channels.size());

  Preprocess(img, &input_channels);

	dbg ("%s","before Forward");
  net_->Forward();
	dbg("%s","after Forward");

  /* Copy the output layer to a std::vector */
  caffe::Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  std::vector<std::vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    std::vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */

void Detector::SetMean(const std::string& mean_file, const std::string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    caffe::BlobProto blob_proto;
    caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    caffe::Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
		dbg ("mean_blob: height=%d,width=%d",mean_blob.height(), mean_blob.width());
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    std::stringstream ss(mean_value);
    std::vector<float> values;
    std::string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
	dbg ("img.channels=%d, num_channels=%d",img.channels(),num_channels_);
	dbg("size=%d,cols=%d,rows=%d",img.size(),img.cols,img.rows);
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
	if (!cfg.color) 
	    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else
    cv::cvtColor(img, sample, cv::COLOR_RGB2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;
#if 0
#ifdef DEBUG_MSR
	cv::namedWindow ("resized");
	cv::imshow("resized",sample_resized);
	cv::waitKey();
	cv::destroyWindow("resized");
#endif
#endif

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

	if (cfg.usemean) {
		cv::Mat sample_normalized;
		cv::subtract(sample_float, mean_, sample_normalized);
		dbg ("mean_ cols/rows=%d/%d, sample_norm cols/rows=%d/%d",
			mean_.cols,mean_.rows,sample_normalized.cols,sample_normalized.rows);
#ifdef DEBUG_MSR
		cv::namedWindow ("normalized");
		cv::imshow("normalized",sample_normalized);
		cv::waitKey();
		cv::destroyWindow("normalized");
	}
#endif
  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {

// if USE_MEAN of config file is false use any string at place of .binaryproto

	if (argc != 8) {
		std::cerr << "Usage: " << argv[0]
			<< " deploy.prototxt network.caffemodel mean.binaryproto"
			<< " labels.txt file.bag secs <config file>" << std::endl;
		return 1;
	}

	const std::string& model_file = argv[1];
	dbg ("prototxt: %s",model_file.c_str());
	const std::string& weights_file = argv[2];
	dbg ("caffemodel: %s",weights_file.c_str());
	const std::string& mean_file = argv[3];
	dbg ("mean_file: %s",mean_file.c_str());
	const std::string& mean_value = "";
	const std::string& label_file   = argv[4];
	dbg ("label: %s\n",label_file.c_str());
	std::string file_type = "image";

	dbg ("config_file = %s",argv[7]);

	struct _config config (argv[7]);
	dbg ("config: usemean=%d,brightness=%d,color=%d,crop=%d",config.usemean,
		config.brightness, config.color, config.crop);

	const float confidence_threshold = 0.49;

	Detector detector(model_file, weights_file, mean_file, mean_value, config);
	std::fstream lf(label_file);
	std::string label;
	std::vector<std::string> labels;
	while (lf >> label)
		labels.push_back (label);
#if 0
	std::string file;
	char fn0 [256];
	char fn1 [256];
	FILE* fl = fopen(argv[4], "r");
	int n;
	while ((n = fscanf(fl, "%s%s",&fn0,&fn1)) != EOF) {
		file = fn0;
		file += " ";
		file += fn1;
		dbg ("image: %s",file.c_str());
//		FILE *fg = fopen (file.c_str(), "r");
//		dbg ("fg=%p");
//		fclose (fg);
// #else
		std::ifstream infile(argv[4]);
		while (infile >> file) {
		dbg ("image: %s",file.c_str());
// #endif
// #else
	std::string fl_png = argv[4];
	ms_cam camera (fl_png);
	dbg ("fl_png=%s",fl_png.c_str());
	dbg ("base_fl=%s, sec_fl=%d",camera.base_fl.c_str(),camera.sec_fl);	
#else
	std::string fl_png = argv[5];
	ms_cam camera (fl_png);
	dbg ("fl_png=%s",fl_png.c_str());
	camera.sec_fl = std::stoi(argv[6]);
#endif // if 0

	if (!camera.ms_cam_skip (camera.sec_fl, config.color)) {
		dbg ("%s","ERROR: ms_cam_skip() false");
		return 1;
	}
	const int w = camera.cam_frame.as<rs2::video_frame>().get_width();
	const int h = camera.cam_frame.as<rs2::video_frame>().get_height();
	cv::Mat img;
	if (!config.color) {
		cv::Mat img1(cv::Size(w, h), CV_8UC1, (void*)camera.cam_frame.get_data(),
			cv::Mat::AUTO_STEP);
		img = img1;
	} else {
		cv::Mat img1(cv::Size(w, h), CV_8UC4, (void*)camera.cam_frame.get_data(),
			cv::Mat::AUTO_STEP);
		img = img1;
	}

	dbg ("image matrix col=%d, row=%d,chammels=%d", img.cols,img.rows,img.channels());

	cv::namedWindow ("bucket");
	cv::imshow("bucket",img);
	cv::waitKey();
	cv::destroyWindow("bucket");

	std::vector<std::vector<float> > detections;
	
	if (config.crop) {
		dbg ("h=%d,w=%d",h,w);
		cv::Rect crop_region(0, h/2, w, h/2);
		cv::Mat img_crop = img(crop_region);

		cv::namedWindow ("Crop");
		cv::imshow("Crop",img_crop);
		cv::waitKey();
		cv::destroyWindow("Crop");

		detections = detector.Detect(img_crop);
	} else {
		if (config.brightness) {
			dbg("img.size=%d,cols=%d,rows=%d,channels=%d",
				img.size(),img.cols, img.rows, img.channels());
			cv::Mat bright_image;

			// double contrast = 1.3; /*< Simple contrast control */
			double contrast = 2.0; /*< Simple contrast control */
			int bright = 90;       /*< Simple brightness control */
			dbg ("contrast=%g,bright=%d",contrast,bright);
			img.convertTo(bright_image, -1, contrast, bright); //increase the brightness by beta
			dbg("bright_image.size=%d,cols=%d,rows=%d,channels=%d",bright_image.size(),
				bright_image.cols, bright_image.rows,bright_image.channels());
			cv::namedWindow ("Bright");
			cv::imshow("Bright",bright_image);
			cv::waitKey();
			cv::destroyWindow("Bright");
			dbg ("%s","zond");

			detections = detector.Detect(bright_image);

		} else
			detections = detector.Detect(img);
	}
	dbg ("detections size = %ld",detections.size());
  	  /* Print the detection results. */
	for (int i = 0; i < detections.size(); ++i) {
		const std::vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		dbg ("d size = %lu", d.size());
		const float score = d[2];
		cv::Point p1;
		p1.x = d[3] * img.cols;
		p1.y = d[4] * img.rows;
		cv::Point p2;
		p2.x = d[5] * img.cols;
		p2.y = d[6] * img.rows;
		dbg ("score=%g,label=%g", score,d[1]);
		if (score >= confidence_threshold) {
			cv::Mat img_orig = img.clone();
			printf ("%s ", fl_png.c_str());
			printf ("%d ", (int)(d[1]));
			printf ("%f ", score);
			printf ("%d ", (int)p1.x);
			printf ("%d ", (int)p1.y);
			printf ("%d ", (int)p2.x);
			printf ("%d\n",  (int)p2.y);
			cv::rectangle(img_orig, p1, p2, {255,255,255});
			char str_text[256];
			int str_len = sprintf (str_text, (const char*)"%s: %.2g", labels[d[1]].c_str(),
				score);
			cv::Point p3 (cv::max(p1.x, 15), cv::max(p1.y, 15));
			std::string str_string (str_text, str_len);
			cv::putText(img_orig, str_string, p3, cv::FONT_ITALIC, 0.6, {255,255,255}, 1);
			cv::namedWindow("SSD");
			cv::imshow("SSD",img_orig);
			int k = cv::waitKey();
			cv::destroyWindow("SSD");
			if (k == 27)	// escape
				break;
		}
	}
	dbg ("end of func %s","here");
	return 0;
}
