using namespace std;
#include <vector>
#include "math.h"
#include <iostream>


struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  float w1;
  float w2;
  float w3;
  float w4;
};

void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    float roi_start_h,
    float roi_start_w,
    float bin_size_h,
    float bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc>* pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const float yy = roi_start_h + ph * bin_size_h +
            static_cast<float>(iy + .5f) * bin_size_h /
                static_cast<float>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const float xx = roi_start_w + pw * bin_size_w +
              static_cast<float>(ix + .5f) * bin_size_w /
                  static_cast<float>(roi_bin_grid_w);

          float x = xx;
          float y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc->at(pre_calc_index) = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = static_cast<int>(y);
          int x_low = static_cast<int>(x);
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (float)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (float)x_low;
          } else {
            x_high = x_low + 1;
          }

          float ly = y - y_low;
          float lx = x - x_low;
          float hy = 1. - ly, hx = 1. - lx;
          float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indeces
          PreCalc pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc->at(pre_calc_index) = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

void ROIAlign(
    const int n_rois,
    const float* bottom_data,
    const float& spatial_scale,
    const bool position_sensitive,
    const bool continuous_coordinate,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const float* bottom_rois,
    int roi_cols,
    float* top_data) {
  // DCHECK(roi_cols == 4 || roi_cols == 5);

  // int n_rois = nthreads / channels / pooled_width / pooled_height;
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
#pragma omp parallel for \
num_threads(12)
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    // roi could have 4 or 5 columns
    const float* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      if (roi_batch_ind < 0) {
        top_data[n] = 0;
        continue;
      }
      offset_bottom_rois++;
    }

    // Do not using rounding; this implementation detail is critical
    float roi_offset = continuous_coordinate ? static_cast<float>(0.5) : static_cast<float>(0);
    float roi_start_w = offset_bottom_rois[0] * spatial_scale - roi_offset;
    float roi_start_h = offset_bottom_rois[1] * spatial_scale - roi_offset;
    float roi_end_w = offset_bottom_rois[2] * spatial_scale - roi_offset;
    float roi_end_h = offset_bottom_rois[3] * spatial_scale - roi_offset;

    float roi_width = roi_end_w - roi_start_w;
    float roi_height = roi_end_h - roi_start_h;
    // if (continuous_coordinate) {
    //   CHECK_GT(roi_width, 0.);
    //   CHECK_GT(roi_height, 0.);
    // } else {  // backward compatiblity
      // Force malformed ROIs to be 1x1
      // roi_width = std::max(roi_width, (T)1.);
      // roi_height = std::max(roi_height, (T)1.);
    // }
    roi_width = std::max(roi_width, (float)1.);
    roi_height = std::max(roi_height, (float)1.);
    float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : std::ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : std::ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const float count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation
    std::vector<PreCalc> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        &pre_calc);

    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          int c_unpooled = c;
          int channels_unpooled = channels;
          if (position_sensitive) {
            c_unpooled = c * pooled_height * pooled_width + ph * pooled_width + pw;
            channels_unpooled = channels * pooled_height * pooled_width;
          }
          const float* offset_bottom_data =
              bottom_data + (roi_batch_ind * channels_unpooled + c_unpooled)
              * height * width;
          float output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc pc = pre_calc[pre_calc_index];
              output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                  pc.w2 * offset_bottom_data[pc.pos2] +
                  pc.w3 * offset_bottom_data[pc.pos3] +
                  pc.w4 * offset_bottom_data[pc.pos4];

              pre_calc_index += 1;
            }
          }
          output_val /= count;

          top_data[index] = output_val;
        }  // for pw
      }  // for ph
    }  // for c
  }  // for n
}

extern "C"{
    void ROIAlignC(const int n_rois,
		   const float* bottom_data,
		   const double spatial_scale,
	           const bool position_sensitive,
	           const bool continuous_coordinate,
	           const int channels,
                   const int height,
	           const int width,
	           const int pooled_height,
	           const int pooled_width,
                   const int sampling_ratio,
	           const float* bottom_rois,
	           int roi_cols,
	           float* top_data){
        float spatial_scale_float = static_cast<float>(spatial_scale);
        ROIAlign(n_rois, bottom_data, spatial_scale, position_sensitive, continuous_coordinate,
                 channels, height, width, pooled_height, pooled_width, sampling_ratio, bottom_rois,
                 roi_cols, top_data);
    }
}
