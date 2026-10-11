/*Copyright (c) 2022 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#pragma once
__device__ const double ptm_template_fcc[1][PTM_MAX_POINTS][3] = {{        {0,0,0},
                                                                {0.70710678118654757,0.70710678118654757,0},
                                                                {0,0.70710678118654757,0.70710678118654757},
                                                                {0.70710678118654757,0,0.70710678118654757},
                                                                {-0.70710678118654757,-0.70710678118654757,0},
                                                                {0,-0.70710678118654757,-0.70710678118654757},
                                                                {-0.70710678118654757,0,-0.70710678118654757},
                                                                {-0.70710678118654757,0.70710678118654757,0},
                                                                {0,-0.70710678118654757,0.70710678118654757},
                                                                {-0.70710678118654757,0,0.70710678118654757},
                                                                {0.70710678118654757,-0.70710678118654757,0},
                                                                {0,0.70710678118654757,-0.70710678118654757},
                                                                {0.70710678118654757,0,-0.70710678118654757},
}};

__device__ const double ptm_template_hcp[2][PTM_MAX_POINTS][3] = {{        {0,0,0},
                                                                {0.5,-0.8660254037844386,0},
                                                                {-1,0,0},
                                                                {-0.5,0.28867513459481287,-0.81649658092772592},
                                                                {0.5,0.28867513459481287,-0.81649658092772592},
                                                                {0,-0.57735026918962573,-0.81649658092772592},
                                                                {-0.5,0.8660254037844386,0},
                                                                {0.5,0.8660254037844386,0},
                                                                {1,0,0},
                                                                {-0.5,-0.8660254037844386,0},
                                                                {0,-0.57735026918962573,0.81649658092772592},
                                                                {0.5,0.28867513459481287,0.81649658092772592},
                                                                {-0.5,0.28867513459481287,0.81649658092772592}, },

                                                        {       {0,0,0},
                                                                {1,0,0},
                                                                {-0.5,-0.8660254037844386,0},
                                                                {-0.5,-0.28867513459481287,-0.81649658092772592},
                                                                {0,0.57735026918962573,-0.81649658092772592},
                                                                {0.5,-0.28867513459481287,-0.81649658092772592},
                                                                {-1,0,0},
                                                                {-0.5,0.8660254037844386,0},
                                                                {0.5,0.8660254037844386,0},
                                                                {0.5,-0.8660254037844386,0},
                                                                {0.5,-0.28867513459481287,0.81649658092772592},
                                                                {0,0.57735026918962573,0.81649658092772592},
                                                                {-0.5,-0.28867513459481287,0.81649658092772592},
}};

__device__ const double ptm_template_bcc[1][PTM_MAX_POINTS][3] = {{        {0,0,0},
                                                                {0.54145188432738056,0.54145188432738056,0.54145188432738056},
                                                                {-0.54145188432738056,0.54145188432738056,0.54145188432738056},
                                                                {0.54145188432738056,0.54145188432738056,-0.54145188432738056},
                                                                {-0.54145188432738056,-0.54145188432738056,0.54145188432738056},
                                                                {0.54145188432738056,-0.54145188432738056,0.54145188432738056},
                                                                {-0.54145188432738056,0.54145188432738056,-0.54145188432738056},
                                                                {-0.54145188432738056,-0.54145188432738056,-0.54145188432738056},
                                                                {0.54145188432738056,-0.54145188432738056,-0.54145188432738056},
                                                                {1.0829037686547611,0,0},
                                                                {-1.0829037686547611,0,0},
                                                                {0,1.0829037686547611,0},
                                                                {0,-1.0829037686547611,0},
                                                                {0,0,1.0829037686547611},
                                                                {0,0,-1.0829037686547611},
}};
