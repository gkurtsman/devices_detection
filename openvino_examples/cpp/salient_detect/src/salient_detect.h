// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#ifdef DEBUG_MSR
#define dbg(format, ...) \
do { \
    printf("%s#%d:" format "\n", __FILENAME__,__LINE__, __VA_ARGS__); \
} while (0)
#else
#define dbg(format, arg...)
#endif

#define SALIENT

enum param_keys {noop,h, m, i, l, c, d};

#define NON_OPTIONAL_PARAMS	2
struct param_vals {
	enum param_keys parm_h;
	enum param_keys parm_m;
	std::string modelPath;
	enum param_keys parm_i;
	std::vector<std::string> images;
	enum param_keys parm_l;
	std::string cpu_plugin_layers;
	enum param_keys parm_c;
	std::string gpu_config;
	enum param_keys parm_d;
	std::string device;
	param_vals ()
	{
		modelPath = "";
		if (!images.empty())
			images.erase(images.begin(), images.end());
		cpu_plugin_layers = "";
		gpu_config = "";
		device = "";
		parm_h=noop;
		parm_m=noop;
		parm_i=noop;
		parm_l=noop;
		parm_c=noop;
		parm_d=noop;
	}
};

/* thickness of a line (in pixels) to be used for bounding boxes */
#define BBOX_THICKNESS 2

#define SIDE_LINE_DEVIATION	5

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model.";

/// @brief message for images argument
static const char image_message[] = "Required. Path to an image.";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown "
                                            "below). "
                                            "Default value is CPU. Use \"-d HETERO:<comma_separated_devices_list>\" format to specify "
                                            "HETERO plugin. "
                                            "Sample will look for a suitable plugin for device specified.";

/// @brief message for plugin custom kernels desc
static const char custom_plugin_cfg_message[] = "Required for GPU, MYRIAD, HDDL custom kernels. "
                                                "Absolute path to the .xml config file with the kernels descriptions.";

/// @brief message for user library argument
static const char custom_ex_library_message[] = "Required for CPU plugin custom layers. "
                                                "Absolute path to a shared library with the kernels implementations.";

/// @brief message for config argument
static constexpr char config_message[] = "Path to the configuration file.";

/**
 * \brief This function show a help message
 */
static void showUsage(std::string pname) {
    std::cout << std::endl;
    std::cout << pname.c_str() << "[OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -m \"<path>\"             " << model_message << std::endl;
    std::cout << "    -i \"<path>\"             " << image_message << std::endl;
	std::cout << std::endl;
    std::cout << "    -l \"<absolute_path>\"  " << custom_ex_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "    -c \"<absolute_path>\"  " << custom_plugin_cfg_message << std::endl;
	std::cout << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
}

struct object_rectangle
{
public:
	float xmin;
	float xmax;
	float ymin;
	float ymax;
	float real_xmin;
	float real_xmax;
	float real_ymin;
	float real_ymax;
	float imageWidth;
	float imageHeight;

	object_rectangle(float image_width, float image_height);
};

void scan_rects (std::vector<struct object_rectangle>& obre_vec, cv::Mat img,
			 int line, int& col_left);
#if 0
inline void showAvailableDevices() {
	InferenceEngine::Core ie;
	std::vector<std::string> devices = ie.GetAvailableDevices();

	std::cout << std::endl;
	std::cout << "Available target devices:";
	for (const auto& device : devices) {
		std::cout << "  " << device;
	}
	std::cout << std::endl;
}

#endif
