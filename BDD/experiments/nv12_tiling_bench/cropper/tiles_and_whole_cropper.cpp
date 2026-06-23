/**
 * Custom hailocropper crop function: emit an N x M uniform grid of tiles PLUS one
 * whole-frame crop. Feeding these N*M+1 crops through a SINGLE hailocropper ->
 * SINGLE hailonet -> SINGLE hailoaggregator runs the "tiles + whole-frame" workload
 * in ONE net-group, avoiding the round-robin two-hailonet scheduler tax (which is
 * frame-fair, not cost-fair, and starves the tiling branch — see
 * BDD/experiments/nv12_tiling_bench/RESULTS.md).
 *
 * Grid size comes from env TILES_X / TILES_Y (the hailocropper crop-function
 * signature takes no parameters). Defaults to 2x2.
 */
#include "tiles_and_whole_cropper.hpp"
#include <cstdlib>
#include <memory>

static int env_int(const char *key, int dflt)
{
    const char *v = std::getenv(key);
    if (!v || !*v)
        return dflt;
    int n = std::atoi(v);
    return n > 0 ? n : dflt;
}

std::vector<HailoROIPtr> create_crops(std::shared_ptr<HailoMat> image, HailoROIPtr roi)
{
    (void)image;
    (void)roi;
    const int nx = env_int("TILES_X", 2);
    const int ny = env_int("TILES_Y", 2);

    std::vector<HailoROIPtr> crops;
    crops.reserve(static_cast<size_t>(nx) * ny + 1);

    // The "+1": whole-frame view (full-image context the tiles lack).
    crops.emplace_back(std::make_shared<HailoROI>(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));

    // N x M uniform grid of tiles (normalized coords). hailocropper resizes each
    // crop to the HEF input (letterboxed); hailoaggregator scales detections back.
    const float w = 1.0f / static_cast<float>(nx);
    const float h = 1.0f / static_cast<float>(ny);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            crops.emplace_back(std::make_shared<HailoROI>(
                HailoBBox(static_cast<float>(i) * w, static_cast<float>(j) * h, w, h)));

    return crops;
}
