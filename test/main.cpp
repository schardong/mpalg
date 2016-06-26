#include "mp.h"
#include <iostream>
#include <vector>

using cv::Mat;

int test_cmdscale()
{
  float d[4][4] = {
    {0, 93, 82, 133},
    {93, 0, 52, 60},
    {82, 52, 0, 111},
    {133, 60, 111, 0}
  };

  Mat dist = Mat(4, 4, CV_32FC1, d);
  Mat evals;
  Mat evecs;
  Mat P = cmdscale(dist, 2, &evals, &evecs);
  return 0;
}

int test_lamp()
{
  float d[10][3] = {
    {1,2,3},
    {3,2,1},
    {4,5,6},
    {2,3,4},
    {4,3,2},
    {6,8,9},
    {2,4,6},
    {1,9,5},
    {1,3,9},
    {9,8,7},
  };

  std::vector<int> idx = {1, 5, 9};
  
  float ys[3][2] = {
    {4, 9},
    {1, 1},
    {9, 6},
  };

  Mat X = Mat(10, 3, CV_32FC1, d);
  Mat Ys = Mat(3, 2, CV_32FC1, ys);

  Mat P = lamp(X, idx, Ys);
  return 0;
}

int main()
{
  test_cmdscale();
  test_lamp();
  return 0;
}
