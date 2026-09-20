"""Versioned representation layout and explicit query/view contracts."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Layout:
    invariant_dim: int = 128
    degrees: tuple = (1, 2, 4, 6)
    channels: int = 4
    radii: tuple = (3.5, 5., 6.5, 8.)
    basis: str = 'e3nn real component-normalized solid harmonics'
    parity: str = '(-1)**ell; O(3)'
    role: str = 'all equivariant channels directly supervised against fixed moments'
    scale: str = 'one training-RMS scalar per degree/radius, shared by m; floor 1e-3'

    @property
    def widths(self):
        return tuple(self.channels*(2*l+1) for l in self.degrees)

    @property
    def equivariant_dim(self):
        return sum(self.widths)

    @property
    def packed_dim(self):
        return self.invariant_dim+self.equivariant_dim

    def metadata(self):
        return asdict(self)


LAYOUT = Layout()


@dataclass(frozen=True)
class Query:
    family: str
    time_index: int
    neighbor_index: int
    history_allowed: bool = False


def queries(spec):
    result = []
    for family, time_index in [('present_neighbors', 1), ('future_neighbors', 2)]:
        if spec['family_weights'][family] > 0:
            result.extend(Query(family, time_index, j) for j in range(1, spec['neighbors']+1))
    if spec['family_weights']['future_center'] > 0 or spec['future_weight'] > 0:
        result.append(Query('future_center', 2, 0))
    return tuple(result)


@dataclass(frozen=True)
class RequiredViewPlan:
    views: tuple
    query_list: tuple

    @classmethod
    def from_spec(cls, spec, all_views=False):
        q = queries(spec)
        required = {(1, 0), (2, 0)}
        required.update((x.time_index, x.neighbor_index) for x in q
                        if spec['family_weights'][x.family] > 0)
        if all_views:
            required = {(t, j) for t in range(3) for j in range(spec['neighbors']+1)}
        return cls(tuple(sorted(required)), q)

    def slot(self, time_index, neighbor_index):
        return self.views.index((time_index, neighbor_index))


def variants(config):
    """One common parameter initialization and equal anchor/update budget across A–E."""
    result = []
    for arm in 'ABCDE':
        result.append(dict(name=f'{arm}-seed{config["seed"]}', arm=arm, neighbors=6,
            family_weights=dict(present_neighbors=float(arm in 'CE'),
                                future_neighbors=float(arm in 'DE'),
                                future_center=float(arm in 'DE')),
            future_weight=0. if arm == 'A' else .25,
            prediction_weight=.1, geometry_weight=.1,
            sigreg_weight=.1, sigreg_mode='per_sample_discrepancy',
            encoder_lr=.0002, head_lr=.002))
    return result
