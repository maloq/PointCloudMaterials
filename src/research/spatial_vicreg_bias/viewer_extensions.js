function latticeOverlays(patch,fit,color){
  const result=[],trace=(points,extra)=>({type:'scatter3d',x:points.map(v=>v?.[0]??null),y:points.map(v=>v?.[1]??null),z:points.map(v=>v?.[2]??null),hoverinfo:'skip',showlegend:false,...extra});
  if(!fit.grid.length)return result;
  if(el('sampleStructure').checked){const shell=fit.shell.filter(i=>i!==0).map(i=>patch.xyz[i]);result.push({type:'mesh3d',x:shell.map(v=>v[0]),y:shell.map(v=>v[1]),z:shell.map(v=>v[2]),alphahull:0,opacity:.15,color,hoverinfo:'skip',showlegend:false});}
  if(el('sampleGrid').checked){
    const matched=new Set(fit.matched),sites=fit.matched.map(i=>fit.grid[i]);
    const edges=fit.edges.filter(([a,b])=>matched.has(a)&&matched.has(b));
    result.push(trace(edges.flatMap(([a,b])=>[fit.grid[a],fit.grid[b],null]),{mode:'lines',line:{color:'#75869a',width:1},opacity:.18}),trace(sites,{mode:'markers',marker:{size:2.5,symbol:'circle-open',color:'#34465c'},opacity:.55}));
  }
  if(el('sampleResidual').checked)result.push(trace(patch.xyz.flatMap((p,i)=>[p,fit.grid[fit.matched[i]],null]),{mode:'lines',line:{color:'#111827',width:3}}));
  return result;
}
let travelData=null,travelIdentity='',travelRevision=0,activePath=[],travelDisplayKey='',travelProfile=null,pathRevision=0;
function vectorNorm(v){return Math.hypot(...v);}
function spatialDelta(a,b,box){return a.map((v,j)=>{let d=v-b[j];return box?d-Math.round(d/box[j])*box[j]:d;});}
function vectorDelta(a,b){return a.map((v,j)=>v-b[j]);}
async function prepareTravel(){
  const revision=++travelRevision,p=pair(),snapshot=el('mdSnapshot').value,identity=snapshot+'/'+p.nn.id;
  const displayKey=JSON.stringify([identity,p.descriptor.id,el('color').value,p.map]);
  if(identity===travelIdentity){if(displayKey!==travelDisplayKey){travelDisplayKey=displayKey;for(const id of ['pathFrom','pathTo'])for(const option of el(id).options)option.text=label(Number(option.value),true,p.map);if(activePath.length)await renderTravel();}return;}
  el('embeddingTravel').hidden=false;el('pathStatus').textContent='Loading embedding path observations…';
  const entry=D.travel[snapshot],model=entry.models[p.nn.id];
  await Promise.all([loadAsset(entry.geometry.asset),loadAsset(model.asset)]);if(revision!==travelRevision)return;
  travelData={geometry:window.TRAVEL_GEOMETRY[entry.geometry.key],features:window.TRAVEL_EMBEDDINGS[model.key]};travelIdentity=identity;travelDisplayKey=displayKey;activePath=[];travelProfile=null;++pathRevision;
  pathWorker.postMessage({type:'data',identity,geometry:travelData.geometry,features:travelData.features});
  const present=[...new Set(travelData.features.clusters)].sort((a,b)=>p.map[a]-p.map[b]);
  for(const id of ['pathFrom','pathTo']){const old=el(id).value;el(id).replaceChildren();for(const c of present)el(id).add(new Option(label(c,true,p.map),c));if(present.includes(Number(old))&&old!=='')el(id).value=old;else el(id).value=id==='pathFrom'?present[0]:present.at(-1);}
  await buildTravel();
}
// Dedicated worker keeps graph construction and shortest paths off the UI thread.
function pathWorkerMain(){
 let identity='',geometry,features;const graphs=new Map();
 const norm=v=>Math.hypot(...v),delta=(a,b,box)=>a.map((v,j)=>{let d=v-b[j];return box?d-Math.round(d/box[j])*box[j]:d;});
 function graph(kind){
  if(graphs.has(kind))return graphs.get(kind);
  const spatial=kind==='spatial',neighbors=spatial?geometry.neighbors:features.neighbors,values=spatial?geometry.xyz:features.z;
  const links=Array.from({length:neighbors.length},()=>new Set());neighbors.forEach((rows,i)=>rows.forEach(j=>{links[i].add(j);links[j].add(i);}));
  const weights=new Map(),n=links.length;
  const result=links.map((rows,i)=>[...rows].map(j=>{const key=Math.min(i,j)*n+Math.max(i,j);if(!weights.has(key))weights.set(key,norm(delta(values[Math.min(i,j)],values[Math.max(i,j)],spatial?geometry.box:null)));return [j,weights.get(key)];}));
  graphs.set(kind,result);return result;
 }
 function route(start,end,links){
  const n=links.length,dist=new Float64Array(n).fill(Infinity),prev=new Int32Array(n).fill(-1),heap=[];
  const less=(a,b)=>a[0]<b[0]||(a[0]===b[0]&&a[1]<b[1]);
  function push(value){let i=heap.length;heap.push(value);while(i){let parent=(i-1)>>1;if(!less(value,heap[parent]))break;heap[i]=heap[parent];i=parent;}heap[i]=value;}
  function pop(){const first=heap[0],last=heap.pop();if(heap.length){let i=0;while(2*i+1<heap.length){let j=2*i+1;if(j+1<heap.length&&less(heap[j+1],heap[j]))++j;if(!less(heap[j],last))break;heap[i]=heap[j];i=j;}heap[i]=last;}return first;}
  dist[start]=0;push([0,start]);
  while(heap.length){const [value,u]=pop();if(value!==dist[u])continue;if(u===end){const path=[];for(let v=end;v!==-1;v=prev[v])path.push(v);return path.reverse();}
   for(const [v,weight] of links[u]){const candidate=value+weight;if(candidate<dist[v]){dist[v]=candidate;prev[v]=u;push([candidate,v]);}}
  }return [];
 }
 self.onmessage=({data:m})=>{
  try{if(m.type==='data'){identity=m.identity;geometry=m.geometry;features=m.features;graphs.clear();return;}
   if(m.identity!==identity)throw new Error('Path worker observation identity mismatch');
   self.postMessage({id:m.id,path:route(m.start,m.end,graph(m.kind))});
  }catch(error){self.postMessage({id:m.id,error:error.message});}
 };
}
const pathWorkerURL=URL.createObjectURL(new Blob(['('+pathWorkerMain.toString()+')()'],{type:'text/javascript'}));
const pathWorker=new Worker(pathWorkerURL);URL.revokeObjectURL(pathWorkerURL);
const pathRequests=new Map();let pathRequestID=0;
pathWorker.onmessage=({data:m})=>{const pending=pathRequests.get(m.id);if(!pending)return;pathRequests.delete(m.id);if(m.error)pending.reject(new Error(m.error));else pending.resolve(m.path);};
pathWorker.onerror=event=>{for(const pending of pathRequests.values())pending.reject(new Error(event.message));pathRequests.clear();};
function findPath(start,end,kind){return new Promise((resolve,reject)=>{const id=++pathRequestID;pathRequests.set(id,{resolve,reject});pathWorker.postMessage({type:'route',id,identity:travelIdentity,start,end,kind});});}
async function buildTravel(){
  if(!travelData)return;const revision=++pathRevision,identity=travelIdentity;const {geometry:g,features:f}=travelData,from=Number(el('pathFrom').value),to=Number(el('pathTo').value),example=Number(el('pathExample').value);
  const starts=f.clusters.map((c,i)=>c===from?i:-1).filter(i=>i>=0),ends=f.clusters.map((c,i)=>c===to?i:-1).filter(i=>i>=0);
  const start=starts[Math.floor(example*(starts.length-1)/9)];
  const ranked=ends.filter(i=>i!==start).sort((a,b)=>vectorNorm(spatialDelta(g.xyz[b],g.xyz[start],g.box))-vectorNorm(spatialDelta(g.xyz[a],g.xyz[start],g.box)));
  if(!ranked.length){activePath=[];el('pathStatus').textContent='This cluster has only one sampled atom; choose another endpoint.';return;}
  const end=ranked[Math.floor(example*(ranked.length-1)/18)],spatial=el('pathKind').value==='spatial';
  el('pathStatus').textContent='Finding path…';
  const path=await findPath(start,end,spatial?'spatial':'embedding');if(revision!==pathRevision||identity!==travelIdentity)return;activePath=path;
  el('pathStep').max=Math.max(0,activePath.length-1);el('pathStep').value=0;
  if(!activePath.length){el('pathStatus').textContent='No path in the 16-neighbor graph between these sampled endpoints.';await Promise.all(['pathMD','pathProfile','pathCoordinates'].map(id=>Plotly.react(id,[],common,config)));return;}
  await renderTravel();
}
async function renderTravel(){
  if(!activePath.length)return;const {geometry:g,features:f}=travelData,p=pair(),path=activePath,step=Number(el('pathStep').value),spatial=[0],embedding=[0],jump=[0],fromStart=[];
  const xyz=path.map(i=>g.xyz[i]);
  for(let i=0;i<path.length;i++){
    fromStart.push(vectorNorm(vectorDelta(f.z[path[i]],f.z[path[0]])));
    if(i){const delta=spatialDelta(g.xyz[path[i]],g.xyz[path[i-1]],g.box);spatial.push(spatial.at(-1)+vectorNorm(delta));jump.push(vectorNorm(vectorDelta(f.z[path[i]],f.z[path[i-1]])));embedding.push(embedding.at(-1)+jump.at(-1));}
  }
  travelProfile={xyz,spatial,path};
  const colors=path.map(i=>clusterColor(f.clusters[i],true,p.map)),x=spatial,selected=xyz[step];
  el('pathStatus').textContent=path.length+' observed environments · '+g.xyz.length.toLocaleString()+' sampled atoms in snapshot · spatial length '+spatial.at(-1).toFixed(1)+' Å · embedding path length '+embedding.at(-1).toFixed(3)+(g.box?' · original MD cell; lines break at periodic crossings':'');
  el('pathStepLabel').textContent=(step+1)+' / '+path.length+' · atom '+g.atoms[path[step]]+' · '+label(f.clusters[path[step]],true,p.map);
  const spatialTrace={type:'scatter3d',mode:'markers',x:xyz.map(v=>v[0]),y:xyz.map(v=>v[1]),z:xyz.map(v=>v[2]),line:{color:'#677587',width:3},marker:{color:colors,size:6},customdata:path.map((v,i)=>[g.atoms[v],f.clusters[v],i+1]),hovertemplate:'Atom %{customdata[0]} · N%{customdata[1]}<br>Step %{customdata[2]}<extra></extra>'};
  // Markers and context share original coordinates. Never unwrap only one layer.
  const contextXYZ=g.xyz,lineXYZ=[];
  for(let i=1;i<xyz.length;i++){
    const crosses=g.box&&xyz[i].some((v,j)=>Math.abs(v-xyz[i-1][j])>g.box[j]/2);
    if(!crosses)lineXYZ.push(xyz[i-1],xyz[i],null);
  }
  const pathLines={type:'scatter3d',mode:'lines',x:lineXYZ.map(v=>v?.[0]??null),y:lineXYZ.map(v=>v?.[1]??null),z:lineXYZ.map(v=>v?.[2]??null),line:{color:'#677587',width:3},hoverinfo:'skip'};
  const bounds=g.box?g.box.map(length=>[0,length]):Array.from({length:3},(_,j)=>[Math.min(...g.xyz.map(v=>v[j])),Math.max(...g.xyz.map(v=>v[j]))]);
  const mdAxis=j=>({title:{text:['x (Å)','y (Å)','z (Å)'][j]},range:bounds[j]});
  const distanceAxis={title:{text:'Cumulative MD distance (Å)',standoff:16},automargin:true};
  const contextTrace={type:'scatter3d',mode:'markers',x:contextXYZ.map(v=>v[0]),y:contextXYZ.map(v=>v[1]),z:contextXYZ.map(v=>v[2]),marker:{color:f.clusters.map(c=>clusterColor(c,true,p.map)),size:2,opacity:.13},hoverinfo:'skip'};
  const cursor={type:'scatter3d',mode:'markers',x:[selected[0]],y:[selected[1]],z:[selected[2]],marker:{color:'#111827',size:11,symbol:'diamond'},hoverinfo:'skip'};
  const profile=[{type:'scatter',mode:'lines+markers',name:'Distance from start',x,y:fromStart,line:{color:'#35577f'},marker:{color:colors,size:7}},{type:'scatter',mode:'lines',name:'Change from previous atom',x,y:jump,line:{color:'#b55c15'}}];
  const dimensions=f.z[0].length,heat=Array.from({length:dimensions},(_,j)=>path.map(i=>f.z[i][j]));
  const cursorShape={type:'line',x0:x[step],x1:x[step],yref:'paper',y0:0,y1:1,line:{color:'#111827',width:1,dash:'dot'}};
  await Promise.all([
    Plotly.react('pathMD',[contextTrace,spatialTrace,cursor,pathLines,boxTrace(bounds)],{...common,margin:{t:8,b:30,l:8,r:8},showlegend:false,scene:{aspectmode:'data',uirevision:travelIdentity,xaxis:mdAxis(0),yaxis:mdAxis(1),zaxis:mdAxis(2)}},config),
    Plotly.react('pathProfile',profile,{...common,showlegend:false,margin:{t:16,b:72,l:84,r:24},xaxis:distanceAxis,yaxis:{title:{text:'Euclidean embedding distance',standoff:18},automargin:true},shapes:[cursorShape]},config),
    Plotly.react('pathCoordinates',[{type:'heatmap',x,y:Array.from({length:dimensions},(_,i)=>i),z:heat,colorscale:'RdBu',zmid:0,colorbar:{title:{text:'Value'}},hovertemplate:'MD distance %{x:.2f} Å<br>Coordinate %{y}<br>Value %{z:.4f}<extra></extra>'}],{...common,margin:{t:16,b:72,l:72,r:88},xaxis:distanceAxis,yaxis:{title:{text:'Coordinate index',standoff:12},automargin:true},shapes:[cursorShape]},config)]);
}
let cursorPending=false,cursorRunning=null;
function updateTravelCursor(){
 cursorPending=true;if(cursorRunning)return cursorRunning;
 cursorRunning=(async()=>{await new Promise(resolve=>requestAnimationFrame(resolve));while(cursorPending){
  cursorPending=false;if(!travelProfile||!activePath.length)continue;
  const {xyz,spatial,path}=travelProfile,step=Number(el('pathStep').value),point=xyz[step],p=pair();
  el('pathStepLabel').textContent=(step+1)+' / '+path.length+' · atom '+travelData.geometry.atoms[path[step]]+' · '+label(travelData.features.clusters[path[step]],true,p.map);
  const marker={'shapes[0].x0':spatial[step],'shapes[0].x1':spatial[step]};
  await Promise.all([Plotly.restyle('pathMD',{x:[[point[0]]],y:[[point[1]]],z:[[point[2]]]},[2]),Plotly.relayout('pathProfile',marker),Plotly.relayout('pathCoordinates',marker)]);
 }})().finally(()=>{cursorRunning=null;});return cursorRunning;
}
document.addEventListener('DOMContentLoaded',()=>{
  ['pathKind','pathFrom','pathTo','pathExample'].forEach(id=>document.getElementById(id).addEventListener('change',buildTravel));
  document.getElementById('pathBuild').addEventListener('click',buildTravel);
  document.getElementById('pathStep').addEventListener('input',updateTravelCursor);
});
