/**
 * @file mp.h
 * @author Guilherme G. Schardong [gschardong@inf.puc-rio.br]
 * @date 26/06/2016
 * @brief Defitions of the Multidimensional Projection algorithms implemented in
 * this library.
 */

#ifndef MULTIDIMENSIONAL_PROJECTIONS_H
#define MULTIDIMENSIONAL_PROJECTIONS_H

#include <opencv2/core/core.hpp>
#include <vector>

cv::Mat cmdscale(const cv::Mat& dist, const int k,
                 const cv::Mat* eigenvals = NULL,
                 const cv::Mat* eigenvecs = NULL);

cv::Mat lamp(const cv::Mat& X, const std::vector<int> cp_index,
             const cv::Mat& Ys);

#endif // MULTIDIMENSIONAL_PROJECTIONS_H
