# Numerical reference and licenses

The CUDA l=6 recurrence is adapted from `fsph::PointSPHEvaluator<float>` at
https://github.com/glotzerlab/fsph/tree/64d6f81db5e1f29eef565dcd02b81d4ecd5ab056 .
Periodic wrapping and query-image arithmetic follow freud 3.4.0 `Box.h`,
`utils.h`, and `AABBQuery.cc`; normalization and m ordering follow `Steinhardt.cc`:
https://github.com/glotzerlab/freud/tree/v3.4.0 .
This is an independently compiled CUDA backend, not a guarantee of bitwise
libm equivalence. The benchmark records all classification disagreements.

## fsph license

Copyright (c) 2016 The Regents of the University of Michigan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## freud license

BSD 3-Clause License for freud

Copyright (c) 2010-2025 The Regents of the University of Michigan
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## CUDA PTM core

The `ptm/` matcher is adapted from OVITO 3.15.3, commit
`7e9272d91e22f44667bd9a8f65a29c853adc5144`,
[src/3rdparty/ptm](https://gitlab.com/stuko/ovito/-/tree/7e9272d91e22f44667bd9a8f65a29c853adc5144/src/3rdparty/ptm).
The upstream PTM code is MIT licensed, Copyright (c) 2022 PM Larsen.
The QCP rotation code retains the original BSD notice of Pu Liu and Douglas
L. Theobald in `ptm_polar.cuh`. Upstream checksums and the adaptation record
are in `ptm/PROVENANCE.json`; licenses/notices are retained in the source.

Changes: CUDA device annotations and reference-table initialization; only the
FCC/HCP/BCC classification/RMSD path is retained. Neighbor search and bounded
Voronoi-plane clipping are implemented in `ptm_kernels.cu`. They are a separate
implementation, not a bitwise-equivalence claim for all OVITO inputs. The
optional distance ordering is an explicitly approximate PTM variant.
