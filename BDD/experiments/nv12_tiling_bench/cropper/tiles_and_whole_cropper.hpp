#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "hailomat.hpp"

__BEGIN_DECLS
std::vector<HailoROIPtr> create_crops(std::shared_ptr<HailoMat> image, HailoROIPtr roi);
__END_DECLS
