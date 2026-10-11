# Four Ni birth histories and visualization

This is a four-example descriptive analysis, not a Ni catalogue, new simulation,
training cohort or nucleation-rate estimate. Sources are the completed independent
1100 K melts 00, 01, 02 and extension melt04 from the October 7 Ni collections.
Original source IDs, roles, potential identity and preparation ancestry are pinned.
Every full input array is verified against its completed binary manifest. Actual
saved integer steps must be 0,100,200,... with the original 1 fs integration step.
No retired Al histories or broad spatial-training collections are used.

Each bounded search starts at the saved validated liquid at time zero and ends
20 ps after the first native 1-ps monitor crossing of 128 atoms, or the source end.
The monitor only chooses the search bound. Every saved 0.1 ps frame in that prefix
is independently classified on the full periodic cell with PTM RMSD cutoff 0.1.
FCC/HCP/BCC atoms connect within 3.2 Å using the exact freud neighbor query and
component producer used by the Ni search. The four-atom component minimum and
the unchanged descendant-family tracker define eight-atom embryo appearances.
Establishment requires one connected descendant of at least 64 atoms for 1.5 ps;
disconnected fragments are not summed. Splits, mergers, absorption, uncertain
identity and prefix-end censoring remain in the tracker outputs.

One root family with origin `emergence_from_liquid` and its own confirmed
establishment is selected per source: earliest confirmation, then earliest
embryo onset, then family ID. A missing eligible birth is an explicit failure;
the threshold is never relaxed to obtain four examples. This uses the existing
Ni operational birth definition, including its recorded origin metadata; it is
not an added prediction-neighborhood filter. These family IDs are analysis
references, not entries in a consolidated Ni catalogue. All other detected
families remain in the bounded search provenance.

Continuous atom movies span 12 ps before the selected embryo onset through 2 ps
after establishment confirmation, clipped only at source start. No intermediate
saved frames are omitted or interpolated. A 14 Å crop follows the founder atom
nearest the recorded embryo centroid. Atom IDs, current PTM types, founder mask,
tracked largest-family membership and retrospective precursor membership are
separate fields. Coordinates for display are float32 local differences of the
original saved float16 positions; no MD integration or relaxation is performed.

Precursor components include sizes 1–3. Backward links retain both membership
overlap fractions and exact shared atom IDs. A strong link shares at least
min(4,parent size,child size) atoms and at least half of either component. A strong
adjacent-frame link is direct; a link over one intervening frame is marked gap.
The intervening coordinates exist: it is the strong identity association that
is absent. Weaker overlaps are contextual and not recursively expanded. All
qualifying branches are retained. The anchor must reproduce founder IDs exactly.
Reaching the 12 ps bound is left censoring, not a first-ever crystallinity claim.
The precursor graph does not revise embryo labels or produce additional births.

Frame metrics follow the Al visualization definitions: FCC/HCP/BCC counts and
fraction in the full crop; center q6 and neighbor-averaged qbar6, q6 neighbor
coherence, and mean first-twelve-neighbor spacing from the nearest80 descriptor
producer. Shape uses visible precursor/family atoms: gyration radius is sqrt(sum
of covariance eigenvalues), anisotropy is (largest-smallest)/sum. Empty selections
are undefined. Founder retention and founder crystalline count are shown
separately. Shapes and visible membership counts do not measure the entire
component when it extends beyond the crop. Family-size histories retain every
recorded point within the searched prefix and preserve tracking interruptions.

The six property panels, precursor graph, interactive rotation/time slider and
two 0.2-ps-stride GIFs per example read these saved packets. The browser movie
retains all native 0.1 ps frames. The in-chat establishment excerpt may reduce
liquid context points and display coordinate precision, but retains every
crystalline atom and actual frame; displayed scientific properties always come
from the complete crop. All future membership is retrospective annotation and
never becomes encoder or predictor input. The full Ni campaign and its separate
queued population search are unchanged.
