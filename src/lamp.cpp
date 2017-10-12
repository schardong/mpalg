/**
 * @file lamp.cpp
 * @author Guilherme G. Schardong [gschardong@inf.puc-rio.br]
 * @date 26/06/2016
 * @brief Implementation of the Local Affine Multidimensional Projection (LAMP)
 * algorithm.
 * ref:
 * http://ieeexplore.ieee.org/xpl/articleDetails.jsp?reload=true&arnumber=6065024
 */

#include "mp.h"
#include <cstdio>
#include <iostream>
#include <opencv2/core/core_c.h>

using cv::Mat;

static bool s_CheckInputErrorsLAMP(const Mat& X,
                                   const std::vector<int> cp_index,
                                   const Mat& Ys)
{
  if (X.rows < 3) {
    fprintf(stderr, "ERROR: s_CheckInputErrorsLAMP - Input matrix too small to "
                    "execute the projection [%d, %d]\n",
            X.rows, X.cols);
    return false;
  }

  if (cp_index.size() <= 1) {
    fprintf(stderr, "ERROR: s_CheckInputErrorsLAMP - To few control points. "
                    "Must have at least 2.\n");
    return false;
  }

  if (Ys.rows != static_cast<int>(cp_index.size())) {
    fprintf(stderr, "ERROR: s_CheckInputErrorsLAMP - Number of control points "
                    "and number of control points' projections is different. "
                    "%ld control points given and %d samples provided.\n",
            cp_index.size(), Ys.rows);
    return false;
  }

  if (X.cols < Ys.cols) {
    fprintf(stderr, "ERROR: s_CheckInputErrorsLAMP - Projections have more "
                    "dimensions than original data. Original data has %d "
                    "dimensions and projection data has %d dimensions.\n",
            X.cols, Ys.cols);
    return false;
  }

  return true;
}

Mat lamp(const Mat& X, const std::vector<int> cp_index, const Mat& Ys)
{
  if (!s_CheckInputErrorsLAMP(X, cp_index, Ys)) {
    fprintf(stderr, "ERROR: lamp - Invalid input.\n");
    return Mat::zeros(1, 1, CV_8UC1);
  }

  double tol = 1E-003;

  // Building an array with the indices of the points to be projected.
  std::vector<int> proj_idx(X.rows);
  for (int i = 0; i < X.rows; ++i)
    proj_idx[i] = i;
  for (int i = 0; i < cp_index.size(); ++i)
    proj_idx[cp_index[i]] = -1;

  // Building the control points and projected points matrices.
  Mat Xs = Mat::zeros(cp_index.size(), X.cols, X.depth());
  Mat Y = Mat::zeros(X.rows, Ys.cols, Ys.depth());
  for (int i = 0; i < Xs.rows; ++i) {
    X.row(cp_index[i]).copyTo(Xs.row(i));
    Ys.row(i).copyTo(Y.row(cp_index[i]));
  }

  Mat alpha = Mat::zeros(1, cp_index.size(), CV_32FC1);
  for (int i = 0; i < X.rows; ++i) {
    if (proj_idx[i] == -1)
      continue;

    // Building the weights of each control point over the current point.
    for (int j = 0; j < static_cast<int>(cp_index.size()); ++j)
      alpha.at<float>(0, j) =
          1 / cv::max(cv::norm(Xs.row(j), X.row(proj_idx[i])), tol);

    float sum_alpha = cv::sum(alpha)[0];

    Mat T = Mat::zeros(Xs.rows, Xs.cols, Xs.depth());
    for (int k = 0; k < Xs.cols; ++k)
      T.col(k) = Xs.col(k).mul(alpha.t());

    // Building the x-tilde and y-tilde variables (Eq. 3).
    Mat Xtil;
    cv::reduce(T, Xtil, 0, CV_REDUCE_SUM);
    Xtil = Xtil * (1 / sum_alpha);

    T = Mat::zeros(Ys.rows, Ys.cols, Ys.depth());
    for (int k = 0; k < Ys.cols; ++k)
      T.col(k) = Ys.col(k).mul(alpha.t());

    Mat Ytil;
    cv::reduce(T, Ytil, 0, CV_REDUCE_SUM);
    Ytil = Ytil * (1 / sum_alpha);

    // Building the x-hat and y-hat variables (Eq. 4).
    Mat Xhat = Mat::zeros(Xs.rows, Xs.cols, Xs.depth());
    for (int k = 0; k < Xs.rows; ++k)
      Xhat.row(k) = Xs.row(k) - Xtil;

    Mat Yhat = Mat::zeros(Ys.rows, Ys.cols, Ys.depth());
    for (int k = 0; k < Ys.rows; ++k)
      Yhat.row(k) = Ys.row(k) - Ytil;

    // Building the A and B matrices (Eq. 6) and calculating the SVD of t(A) *
    // B.
    Mat sqrt_alpha;
    cv::sqrt(alpha, sqrt_alpha);
    sqrt_alpha = sqrt_alpha.t();

    Mat A;
    Xhat.copyTo(A);
    for (int k = 0; k < A.cols; ++k)
      A.col(k) = A.col(k).mul(sqrt_alpha);

    Mat B;
    Yhat.copyTo(B);
    for (int k = 0; k < B.cols; k++)
      B.col(k) = B.col(k).mul(sqrt_alpha);

    cv::SVD udv(A.t() * B);

    // Calculating the affine transform matrix (Eq. 7).
    Mat M = udv.u * udv.vt;

    // Projecting X[i] using the matrix M (Eq 8)
    Y.row(proj_idx[i]) = (X.row(proj_idx[i]) - Xtil) * M + Ytil;
  }

std::cout << Y << std::endl;
  return Y;
}
