"""Read birth-centred history supplements without changing catalogue events."""
import json

from src.experiment_runner.artifacts import file_hash as sha,json_digest as digest
from src.postprocessing.datasets.births import connect, read


def history_sets(data):
    path=data.root/'histories/index.json'
    return [] if not path.exists() else [e for e in read(path)['histories'].values() if e['cadence']==data.cadence]


class NucleiBirthHistories:
    def __init__(self,data,name='birth-tails-010-v1',*,expected_fingerprint=None):
        entry=next(e for e in history_sets(data) if e['name']==name)
        self.root=data.root/entry['path'];self.manifest=read(self.root/'manifest.json');self.fingerprint=self.manifest['fingerprint']
        if digest({k:v for k,v in self.manifest.items() if k!='fingerprint'})!=self.fingerprint or entry['fingerprint']!=self.fingerprint:raise ValueError('History manifest identity changed')
        if self.manifest['state']!='complete':raise ValueError('History collection is not sealed')
        if expected_fingerprint is not None and self.fingerprint!=expected_fingerprint:raise ValueError('History fingerprint changed')
        if sha(self.root/'histories.sqlite')!=self.manifest['database_sha256']:raise ValueError('History index changed')
        plan=read(self.root/'provenance/plan.json')
        for s in plan['sources']:
            if digest(data.source(s['source_id']))!=digest(s['source']):raise ValueError('History source changed')
        self.db=connect(self.root/'histories.sqlite',readonly=True)

    def close(self):self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def births(self):
        """Original event records, including explicit physical-view aliases."""
        for row in self.db.execute('SELECT * FROM births ORDER BY event_id'):
            value=json.loads(row['summary_json'])
            value['alias_event_ids']=[r[0] for r in self.db.execute('SELECT event_id FROM births WHERE physical_birth_key=? ORDER BY event_id',(row['physical_birth_key'],))]
            yield value

    def history(self,event_id):
        row=self.db.execute('SELECT path,files_json FROM births WHERE event_id=?',(event_id,)).fetchone()
        if row is None:raise KeyError(event_id)
        folder=self.root/row['path']
        if any(sha(folder/n)!=h for n,h in json.loads(row['files_json']).items()):raise ValueError('History packet changed')
        return dict(birth=read(folder/'birth.json'),precursors=read(folder/'graph.json'),family_trace=read(folder/'family-history.json'),
            movie_frames=read(folder/'movie-frames.json'),movie_path=folder/'movie.npz',observations=list(self.links(event_id)))

    def links(self,event_id):
        for row in self.db.execute('''SELECT l.*,o.source_id,o.frame,o.matched_births FROM links l
            JOIN observations o USING(observation_id,target) WHERE l.event_id=? ORDER BY o.frame''',(event_id,)):
            v=dict(row);v['atom_ids']=json.loads(v.pop('atom_ids_json'));yield v
