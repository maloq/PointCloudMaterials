/*Copyright (c) 2022 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#pragma once
#include "ptm_graph_data.h"
#include "ptm_templates.h"
namespace ptm {
struct refdata_t {
 int type, num_nbrs, num_facets, max_degree, num_graphs;
 graph_t* graphs;
 const double (*points)[PTM_MAX_POINTS][3];
};
__device__ const refdata_t structure_fcc = {1,12,20,6,NUM_FCC_GRAPHS,graphs_fcc,ptm_template_fcc};
__device__ const refdata_t structure_hcp = {2,12,20,6,NUM_HCP_GRAPHS,graphs_hcp,ptm_template_hcp};
__device__ const refdata_t structure_bcc = {3,14,24,8,NUM_BCC_GRAPHS,graphs_bcc,ptm_template_bcc};
}
