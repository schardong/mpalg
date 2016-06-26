/**
 * @file mds.cpp
 * @author Guilherme G. Schardong [gschardong@inf.puc-rio.br]
 * @date 26/06/2016
 * @brief Implementation of the Multidimensional Scaling algorithm.
 */

#include "mp.h"
#include <iostream>

using cv::Mat;

static bool s_CheckInpuErrorsMDS(const Mat& dist)
{
  const char* msg_prefix = "ERROR: s_CheckInpuErrorsMDS -";
  if (dist.rows != dist.cols) {
    fprintf(stderr, "%s Matrix is not square [%d, %d].\n", msg_prefix, dist.rows, dist.cols);
    return false;
  }

  if (dist.rows < 3) {
    fprintf(stderr, "%s Too few samples provided (%d). Must have 3 or more samples.\n", msg_prefix, dist.rows);
    return false;
  }
  
  Mat cmpop = dist != dist.t();
  if (cv::sum(cmpop)[0] != 0) {
    fprintf(stderr, "%s Matrix is not symmetric.\n", msg_prefix);
    return false;
  }
  return true;
}

Mat cmdscale(const Mat& dist, const int k, const Mat* eigenvals, const Mat* eigenvecs)
{
  if (!s_CheckInpuErrorsMDS(dist)) {
    fprintf(stderr, "ERROR: cmdscale - Invalid input.\n");
    return Mat::zeros(1, 1, CV_8UC1);
  }

  int sz = dist.total();
  Mat pow_dist;
  
  cv::pow(dist, 2, pow_dist);
  Mat centering = Mat::eye(dist.cols, dist.cols, dist.depth()) - Mat::ones(dist.cols, dist.cols, dist.depth()) * (1.f / dist.cols);
  Mat B = centering * pow_dist * centering * 0.5f;

  Mat evals;
  Mat evecs;
  cv::eigen(B, evals, evecs);

  if (eigenvals != NULL)
    eigenvals = new Mat(evals);

  if (eigenvecs != NULL)
    eigenvecs = new Mat(evecs);

  Mat sqrt_evals = Mat::diag(evals.rowRange(evals.rows - k, evals.rows));
  cv::sqrt(cv::abs(sqrt_evals), sqrt_evals);
  evecs = evecs.rowRange(evecs.rows - k, evecs.rows);
  Mat mds_points = evecs.t() * sqrt_evals;
  cv::flip(mds_points, mds_points, 1);

  return mds_points;
}
