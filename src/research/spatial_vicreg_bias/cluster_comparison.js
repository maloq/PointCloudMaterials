const el=id=>document.getElementById(id);
const config={responsive:true,displaylogo:false,plotGlPixelRatio:2};
const highlightPointScale=1.4;
const visibilityUID=t=>t.meta?.clusterUID||t.uid;
const clusterVisibility=new Map(),plotKeys=new Map(),plotSizes=new Map();
async function mainPlot(id,makeTraces,layout,key){
 const size=Number(el(id.startsWith('md')?'mdSize':'size').value);
 if(plotKeys.get(id)===key){
  if(plotSizes.get(id)!==size){
   const indices=el(id).data.map((t,i)=>t.mode==='markers'?i:-1).filter(i=>i>=0);
   await Plotly.restyle(id,{'marker.size':indices.map(i=>size*(el(id).data[i].meta?.highlight?highlightPointScale:1))},indices);plotSizes.set(id,size);
  }
  return;
 }
 const traces=makeTraces();
 for(const t of traces)if(t.uid){
  const clusterUID=visibilityUID(t);
  t.meta={...t.meta,clusterUID};
  // Plotly uses uid in CSS selectors when removing traces (e.g. highlights).
  t.uid='t'+Array.from(t.uid,c=>c.codePointAt(0).toString(16)).join('_');
  t.visible=clusterVisibility.get(id+'/'+clusterUID)===false?'legendonly':true;
 }
 await Plotly.react(id,traces,{...layout,showlegend:false},config);plotKeys.set(id,key);plotSizes.set(id,size);
 const legend=el(id+'Legend');legend.replaceChildren();
 for(const [index,t] of traces.entries())if(t.uid&&!t.meta?.highlight&&t.name&&typeof t.marker?.color==='string'){
  const key=id+'/'+visibilityUID(t),button=document.createElement('button'),dot=document.createElement('i');dot.style.background=t.marker.color;button.append(dot,document.createTextNode(t.name));button.setAttribute('aria-pressed',clusterVisibility.get(key)!==false);
  button.onclick=()=>{const visible=clusterVisibility.get(key)===false;clusterVisibility.set(key,visible);button.setAttribute('aria-pressed',visible);Plotly.restyle(id,{visible:visible?true:'legendonly'},traces.map((v,i)=>visibilityUID(v)===visibilityUID(t)?i:-1).filter(i=>i>=0));};legend.append(button);
 }
}
const common={margin:{t:12,b:45,l:45,r:20},paper_bgcolor:'white',plot_bgcolor:'white',legend:{orientation:'h'},uirevision:'fixed'};
if(D.descriptor_fit){
 for(const [id,file] of [['metricDefinitions','METRICS.md'],['metricTable','metrics.csv']])if(el(id))el(id).href=D.descriptor_fit.metrics.replace('METRICS.md',file);
}
const families={'Joint descriptor clusters':'joint','TDA clusters':'tda','Bond-order clusters':'bond_order','CNA clusters':'cna'};
const loaded=new Map(),assetInfo=new Map(),assetAccess=new Map(),assetPending=new Set();let assetClock=0;
function registerAssets(entries,namespace){for(const entry of entries)assetInfo.set(entry.asset,{namespace,key:entry.key});}
registerAssets([...D.paired.neural,...D.paired.descriptors],'PACMAP_LAYOUTS');
registerAssets(D.md.snapshots.map(s=>({...s,key:s.key})),'MD_SNAPSHOTS');
for(const model of D.md.models)registerAssets(Object.values(model.snapshots),'MD_NEURAL');
for(const [field,namespace] of [['samples','CLUSTER_SAMPLES'],['lattice','LATTICE_SAMPLES']])if(D[field])for(const snapshot of Object.values(D[field]))for(const entries of Object.values(snapshot))registerAssets(Object.values(entries),namespace);
if(D.travel)for(const snapshot of Object.values(D.travel)){registerAssets([snapshot.geometry],'TRAVEL_GEOMETRY');registerAssets(Object.values(snapshot.models),'TRAVEL_EMBEDDINGS');}
function trimAssets(){
 const p=pair(),snapshot=el('mdSnapshot').value,family=families[p.descriptor.field],snap=D.md.snapshots.find(s=>s.key===snapshot),model=D.md.models.find(s=>s.id===p.nn.id);
 const keep=new Set([p.nn.asset,p.descriptor.asset,snap.asset,model.snapshots[snapshot].asset,...assetPending]);
 for(const field of ['samples','lattice'])if(D[field]){keep.add(D[field][snapshot].neural[p.nn.id].asset);keep.add(D[field][snapshot].descriptors[family].asset);}
 if(D.travel){keep.add(D.travel[snapshot].geometry.asset);keep.add(D.travel[snapshot].models[p.nn.id].asset);}
 const limits={PACMAP_LAYOUTS:6,MD_SNAPSHOTS:4,MD_NEURAL:8,CLUSTER_SAMPLES:8,LATTICE_SAMPLES:8,TRAVEL_GEOMETRY:3,TRAVEL_EMBEDDINGS:3};
 for(const [namespace,limit] of Object.entries(limits)){
  const urls=[...loaded.keys()].filter(url=>assetInfo.get(url).namespace===namespace).sort((a,b)=>assetAccess.get(a)-assetAccess.get(b));
  let excess=urls.length-limit;
  for(const url of urls)if(excess>0&&!keep.has(url)){delete window[namespace][assetInfo.get(url).key];loaded.delete(url);assetAccess.delete(url);--excess;}
 }
}
function loadAsset(url){
  assetAccess.set(url,++assetClock);
  if(!loaded.has(url))loaded.set(url,new Promise((resolve,reject)=>{
    assetPending.add(url);const script=document.createElement('script');script.src=url;script.onload=()=>{script.remove();assetPending.delete(url);trimAssets();resolve();};
    script.onerror=()=>{script.remove();assetPending.delete(url);loaded.delete(url);reject(new Error('Cannot load '+url));};document.head.appendChild(script);
  }));
  return loaded.get(url);
}
for(const [id,spaces] of [['nnSpace',D.paired.neural],['descriptorSpace',D.paired.descriptors]])
  spaces.forEach(s=>el(id).add(new Option(s.title,s.id)));
el('nnSpace').value=D.paired.default_neural;el('descriptorSpace').value=D.paired.default_descriptor;
[...new Set(D.frame)].sort((a,b)=>a-b).forEach(f=>el('frame').add(new Option(D.explorer?'Frame '+f:f+' ps',f)));
D.md.snapshots.forEach(s=>el('mdSnapshot').add(new Option(D.explorer?'Source '+s.source+' · frame '+s.frame:s.frame+' ps',s.key)));
const frameValues=[...new Set(D.frame)].sort((a,b)=>a-b);
el('frameSlider').max=frameValues.length-1;el('frameSlider').value=0;
el('mdFrameSlider').max=D.md.snapshots.length-1;el('mdFrameSlider').value=0;
el('mdFrameSlider').disabled=D.md.snapshots.length===1;
el('mdFrameControl').hidden=!D.explorer;
const linkedFrames=D.md.snapshots.length>1;
function frameLabels(){
  el('frameLabel').textContent=frameValues[Number(el('frameSlider').value)]+(D.explorer?'':' ps');
  const snapshot=D.md.snapshots[Number(el('mdFrameSlider').value)];
  el('mdFrameLabel').textContent=D.explorer?'Source '+snapshot.source+' · frame '+snapshot.frame+(linkedFrames?'':' (single snapshot)'):snapshot.frame+' ps';
}
let frameTimer,frameRender=Promise.resolve(),frameSequence=0;
function slideFrame(fromMD){
  el('allFrames').checked=false;
  if(fromMD){
    const snap=D.md.snapshots[Number(el('mdFrameSlider').value)];el('mdSnapshot').value=snap.key;
    if(linkedFrames){el('frameSlider').value=frameValues.indexOf(snap.frame);el('frame').value=snap.frame;}
  }else{
    const frame=frameValues[Number(el('frameSlider').value)];el('frame').value=frame;
    if(linkedFrames){const i=D.md.snapshots.findIndex(s=>s.frame===frame);if(i<0)throw new Error('Missing full MD frame '+frame);el('mdFrameSlider').value=i;el('mdSnapshot').value=D.md.snapshots[i].key;}
  }
  frameLabels();clearTimeout(frameTimer);const sequence=++frameSequence;
  frameTimer=setTimeout(()=>{if(sequence===frameSequence)frameRender=requestViewer();},80);
}
el('frameSlider').addEventListener('input',()=>slideFrame(false));
el('mdFrameSlider').addEventListener('input',()=>slideFrame(true));
el('allFrames').addEventListener('change',()=>{clearTimeout(frameTimer);++frameSequence;if(el('allFrames').checked){el('frame').value='all';frameRender=requestViewer(['projection','correspondence']);}else slideFrame(false);});
frameLabels();
function explorerQuery(){return new URLSearchParams({model:el('nnSpace').value,descriptor:el('descriptorSpace').value});}
function chooseCheckpoint(){
  const chosen=D.paired.neural.find(s=>s.alpha===Number(el('recipe').value)&&s.seed===Number(el('repeat').value)&&s.epoch===Number(el('checkpoint').value)&&s.representation===el('representation').value);
  if(!chosen)throw new Error('Unavailable checkpoint selection');
  el('nnSpace').value=chosen.id;el('checkpointPath').textContent=chosen.checkpoint;
  el('selection').textContent='GeoFormer · '+el('recipe').selectedOptions[0].text+' · epoch '+chosen.epoch+' · '+el('repeat').selectedOptions[0].text+' · '+el('representation').selectedOptions[0].text;
}
if(D.explorer){
  const query=new URLSearchParams(location.search),requested=query.get('model');
  if(requested){if(!D.paired.neural.some(s=>s.id===requested))throw new Error('Unknown checkpoint: '+requested);el('nnSpace').value=requested;}
  if(query.has('descriptor')){if(!D.paired.descriptors.some(s=>s.id===query.get('descriptor')))throw new Error('Unknown descriptor');el('descriptorSpace').value=query.get('descriptor');}
  if(!D.frozen_model){
    const chosen=D.paired.neural.find(s=>s.id===el('nnSpace').value);
    for(const [control,key] of [['recipe','alpha'],['repeat','seed'],['checkpoint','epoch'],['representation','representation']])el(control).value=chosen[key];
    chooseCheckpoint();
    ['recipe','repeat','checkpoint','representation'].forEach(id=>el(id).addEventListener('change',()=>{chooseCheckpoint();frameRender=requestViewer();}));
  }
  [...new Set(D.source)].sort((a,b)=>a-b).forEach(s=>el('source').add(new Option('Source '+s,s)));
  el('source').addEventListener('change',()=>{frameRender=requestViewer(['projection','correspondence']);});
}
function pair(){
  const nn=D.paired.neural.find(s=>s.id===el('nnSpace').value),descriptor=D.paired.descriptors.find(s=>s.id===el('descriptorSpace').value);
  return {nn,descriptor,map:D.matching[nn.id][families[descriptor.field]].neural_to_descriptor};
}
let rowSelectionKey='',rowSelection=[];
function selectedRows(){
  const key=[el('frame').value,D.explorer?el('source').value:'all'].join('/');
  if(key===rowSelectionKey)return rowSelection;
  const selected=[],frame=el('frame').value;
  for(let i=0;i<D.frame.length;i++){
    if(D.explorer&&el('source').value!=='all'&&D.source[i]!==Number(el('source').value))continue;
    if(frame!=='all'&&D.frame[i]!==Number(frame))continue;
    selected.push(i);
  }
  rowSelectionKey=key;rowSelection=selected;return selected;
}
function label(id,neural,map){return neural?(el('color').value==='original'?'N'+id:'N'+id+' → D'+map[id]):'D'+id;}
function clusterColor(id,neural,map){return palette[neural&&el('color').value!=='original'?map[id]:id];}
function highlightProjection(trace,ids){
  if(!el('highlightInterface').checked)return [trace];
  const selected=ids.map(i=>D.distance[i]!==null&&Number.isFinite(D.distance[i])&&D.distance[i]<=12);
  return [false,true].map(highlight=>{
    const rows=ids.map((_,j)=>j).filter(j=>selected[j]===highlight);
    const marker={...trace.marker};
    if(highlight)Object.assign(marker,{opacity:1,size:trace.marker.size*highlightPointScale,line:{color:'#000000',width:.6}});
    if(Array.isArray(marker.color))marker.color=rows.map(j=>marker.color[j]);
    if(highlight)marker.showscale=false;
    return {...trace,uid:trace.uid?(trace.uid+(highlight?'/highlight':'')):undefined,
      meta:{clusterUID:trace.uid,highlight},x:rows.map(j=>trace.x[j]),y:rows.map(j=>trace.y[j]),z:rows.map(j=>trace.z[j]),marker};
  });
}
function projectionTraces(y,indices,space,neural,map){
  const mode=el('color').value,clusters=mode==='matched'||mode==='original',field=clusters?space.field:mode,values=D.fields[field];
  const continuous=field==='Input crystal fraction';
  function make(ids,name,color){return {type:'scatter3d',mode:'markers',name,x:ids.map(i=>y[i][0]),y:ids.map(i=>y[i][1]),z:ids.map(i=>y[i][2]),
    marker:{size:Number(el('size').value),opacity:.85,color,line:{width:0}},hoverinfo:'skip'};}
  if(continuous){
    const good=indices.filter(i=>values[i]!==null),missing=indices.filter(i=>values[i]===null),t=make(good,field,good.map(i=>values[i]));
    Object.assign(t.marker,{colorscale:'Viridis',cmin:0,cmax:1,showscale:true,colorbar:{thickness:10}});
    return [...highlightProjection(t,good),...(missing.length?highlightProjection(make(missing,'Missing','#999999'),missing):[])];
  }
  const groups=new Map(clusters?Array.from({length:7},(_,i)=>[i,[]]):[]);for(const i of indices){const v=values[i];if(!groups.has(v))groups.set(v,[]);groups.get(v).push(i);}
  return [...groups.keys()].sort((a,b)=>clusters&&neural?map[a]-map[b]:a-b).flatMap(v=>
    highlightProjection({...make(groups.get(v),clusters?label(v,neural,map):String(v),clusters?clusterColor(v,neural,map):palette[v]),uid:space.id+'/'+field+'/'+v},groups.get(v)));
}
const projectionBounds=new Map();
function projectionLayout(id,coordinates){
  if(!projectionBounds.has(id)){
    const bounds=Array.from({length:3},()=>[Infinity,-Infinity]);
    for(const point of coordinates)for(let axis=0;axis<3;axis++){bounds[axis][0]=Math.min(bounds[axis][0],point[axis]);bounds[axis][1]=Math.max(bounds[axis][1],point[axis]);}
    projectionBounds.set(id,bounds.map(([lo,hi])=>[lo-.05*(hi-lo),hi+.05*(hi-lo)]));
  }
  const bounds=projectionBounds.get(id),axis=i=>({title:{text:'PaCMAP '+(i+1)},range:bounds[i]});
  return {...common,margin:{t:4,b:30,l:4,r:4},legend:{orientation:'h',font:{size:10}},scene:{aspectmode:'data',uirevision:id,camera:{eye:{x:1.1,y:1.1,z:1.1}},xaxis:axis(0),yaxis:axis(1),zaxis:axis(2)}};
}
function correspondence(indices,p){
  const t=Array.from({length:7},()=>Array(7).fill(0));
  for(const i of indices)t[D.fields[p.nn.field][i]][D.fields[p.descriptor.field][i]]++;
  const nr=t.map(r=>r.reduce((a,b)=>a+b,0)),nd=Array.from({length:7},(_,d)=>t.reduce((s,r)=>s+r[d],0)),n=indices.length;
  const choose2=v=>v*(v-1)/2,cell=t.flat().reduce((s,v)=>s+choose2(v),0),a=nr.reduce((s,v)=>s+choose2(v),0),b=nd.reduce((s,v)=>s+choose2(v),0);
  let ari=null;if(n>=2){const expected=a*b/choose2(n),denom=(a+b)/2-expected;ari=Math.abs(denom)>1e-12?(cell-expected)/denom:1;}
  const matched=t.reduce((s,r,i)=>s+r[p.map[i]],0);
  const order=Array.from({length:7},(_,i)=>i).sort((a,b)=>p.map[a]-p.map[b]);
  const pairs=order.map(i=>{const d=p.map[i],intersection=t[i][d],union=nr[i]+nd[d]-intersection;
    return {neural:i,descriptor:d,intersection,iou:union?intersection/union:null,neuralFraction:nr[i]?intersection/nr[i]:null,descriptorFraction:nd[d]?intersection/nd[d]:null};});
  return {t,nr,nd,n,matched,ari,order,pairs};
}
async function drawCorrespondence(indices,p){
  const s=correspondence(indices,p),percent=el('matrixScale').value==='fraction';
  el('count').textContent=s.n.toLocaleString();el('overlap').textContent=s.n?(100*s.matched/s.n).toFixed(1)+'%':'—';el('ari').textContent=s.ari===null?'—':s.ari.toFixed(3);
  const rowNames=s.order.map(i=>'N'+i+' → D'+p.map[i]),columns=Array.from({length:7},(_,i)=>'D'+i);
  const z=s.order.map(i=>s.t[i].map(v=>percent?(s.nr[i]?100*v/s.nr[i]:null):v));
  const custom=s.order.map(i=>s.t[i].map((v,d)=>[v,s.nr[i]?100*v/s.nr[i]:null,s.nd[d]?100*v/s.nd[d]:null]));
  const text=s.order.map(i=>s.t[i].map(v=>v?(percent?(100*v/s.nr[i]).toFixed(0)+'%':v.toLocaleString()):''));
  const heat={type:'heatmap',x:columns,y:rowNames,z,text,texttemplate:'%{text}',customdata:custom,
    colorscale:[[0,'#f7f9fc'],[1,'#194c87']],zmin:0,zmax:percent?100:undefined,colorbar:{thickness:10,title:{text:percent?'%':'n'}},
    hovertemplate:'%{y}, %{x}<br>%{customdata[0]:,} centers<br>%{customdata[1]:.1f}% of neural cluster<br>%{customdata[2]:.1f}% of descriptor cluster<extra></extra>'};
  const outlines=s.order.map((_,i)=>({type:'rect',xref:'x',yref:'y',x0:i-.48,x1:i+.48,y0:i-.48,y1:i+.48,line:{color:palette[i],width:2},fillcolor:'rgba(0,0,0,0)'}));
  const bars={type:'bar',orientation:'h',y:rowNames,x:s.pairs.map(v=>v.iou),marker:{color:s.pairs.map(v=>palette[v.descriptor])},
    text:s.pairs.map(v=>v.iou===null?'—':(100*v.iou).toFixed(1)+'%'),textposition:'outside',cliponaxis:false,
    customdata:s.pairs.map(v=>[v.intersection,v.neuralFraction,v.descriptorFraction]),
    hovertemplate:'%{y}<br>IoU %{x:.1%}<br>%{customdata[0]:,} shared centers<br>%{customdata[1]:.1%} of neural cluster<br>%{customdata[2]:.1%} of descriptor cluster<extra></extra>'};
  await Promise.all([Plotly.react('matrix',[heat],{...common,margin:{t:12,b:45,l:100,r:40},xaxis:{title:{text:'Descriptor cluster'}},yaxis:{autorange:'reversed'},shapes:outlines},config),
    Plotly.react('pairOverlap',[bars],{...common,margin:{t:12,b:45,l:100,r:60},showlegend:false,xaxis:{range:[0,1.08],tickformat:'.0%'},yaxis:{autorange:'reversed'}},config)]);
}
let projectionRevision=0;
async function drawProjection(){
  const revision=++projectionRevision,indices=selectedRows(),p=pair();el('status').textContent='Loading…';
  try{
    await Promise.all([loadAsset(p.nn.asset),loadAsset(p.descriptor.asset)]);if(revision!==projectionRevision)return;
    const left=window.PACMAP_LAYOUTS[p.nn.key].y3,right=window.PACMAP_LAYOUTS[p.descriptor.key].y3;
    if(D.explorer)D.fields[p.nn.field]=window.PACMAP_LAYOUTS[p.nn.key].clusters;
    if(left.length!==D.frame.length||right.length!==D.frame.length)throw new Error('Projection identity mismatch');
    await Promise.all([mainPlot('two',()=>projectionTraces(left,indices,p.nn,true,p.map),projectionLayout(p.nn.id,left),JSON.stringify([p.nn.id,rowSelectionKey,el('color').value,p.map,el('highlightInterface').checked])),
      mainPlot('three',()=>projectionTraces(right,indices,p.descriptor,false,p.map),projectionLayout(p.descriptor.id,right),JSON.stringify([p.descriptor.id,rowSelectionKey,el('color').value,el('highlightInterface').checked]))]);
    if(revision!==projectionRevision)return;el('status').textContent=indices.length.toLocaleString()+' / '+D.frame.length.toLocaleString()+' centers · colors fixed across snapshots';
  }catch(error){el('status').textContent=error.message;el('status').classList.add('error');throw error;}
}
function boxTrace(bounds){
  const v=[],x=[],y=[],z=[];for(const a of bounds[0])for(const b of bounds[1])for(const c of bounds[2])v.push([a,b,c]);
  for(let i=0;i<8;i++)for(let j=i+1;j<8;j++)if(v[i].filter((a,k)=>a!==v[j][k]).length===1){x.push(v[i][0],v[j][0],null);y.push(v[i][1],v[j][1],null);z.push(v[i][2],v[j][2],null);}
  return {type:'scatter3d',mode:'lines',x,y,z,line:{color:'#7e8c9b',width:2},showlegend:false,hoverinfo:'skip'};
}
function denseTraces(data,values,indices,neural,map,identity){
  const groups=new Map(Array.from({length:7},(_,i)=>[i,[]]));for(const i of indices){const v=values[i];groups.get(v).push(i);}
  const traces=[...groups.keys()].sort((a,b)=>neural?map[a]-map[b]:a-b).map(v=>{const ids=groups.get(v);return {
    uid:identity+'/'+v,type:'scatter3d',mode:'markers',name:label(v,neural,map),x:ids.map(i=>data.x[i]),y:ids.map(i=>data.y[i]),z:ids.map(i=>data.z[i]),
    marker:{size:Number(el('mdSize').value),color:clusterColor(v,neural,map),opacity:1,line:{width:0}},hoverinfo:'skip'};});
  traces.push(boxTrace(data.bounds));return traces;
}
function denseLayout(data){return {...common,margin:{t:4,b:30,l:4,r:4},legend:{orientation:'h',font:{size:10}},scene:{aspectmode:'data',uirevision:'md-space',camera:{projection:{type:'orthographic'},eye:{x:1.1,y:1.1,z:1.05}},
  xaxis:{title:{text:'x (Å)'},range:data.bounds[0]},yaxis:{title:{text:'y (Å)'},range:data.bounds[1]},zaxis:{title:{text:'z (Å)'},range:data.bounds[2]}}};}
let mdRevision=0;
async function drawMD(){
  const revision=++mdRevision,p=pair(),snap=D.md.snapshots.find(s=>s.key===el('mdSnapshot').value),model=D.md.models.find(m=>m.id===p.nn.id),entry=model.snapshots[snap.key];
  try{
    el('mdStatus').textContent='Loading…';await Promise.all([loadAsset(snap.asset),loadAsset(entry.asset)]);if(revision!==mdRevision)return;
    const data=window.MD_SNAPSHOTS[snap.key];
    if(D.explorer)data.bounds=data.box.map(v=>[0,v]);
    const z=data.bounds[2],lo=z[0]+Math.min(Number(el('zLow').value),Number(el('zHigh').value))/100*(z[1]-z[0]),hi=z[0]+Math.max(Number(el('zLow').value),Number(el('zHigh').value))/100*(z[1]-z[0]);
    const indices=[];for(let i=0;i<data.count;i++)if(data.z[i]>=lo&&data.z[i]<=hi)indices.push(i);
    await Promise.all([mainPlot('mdNN',()=>denseTraces(data,window.MD_NEURAL[entry.key][model.representation],indices,true,p.map,p.nn.id),denseLayout(data),JSON.stringify([snap.key,p.nn.id,lo,hi,el('color').value==='original',p.map])),
      mainPlot('mdFeatures',()=>denseTraces(data,data.fields[p.descriptor.field],indices,false,p.map,p.descriptor.id),denseLayout(data),JSON.stringify([snap.key,p.descriptor.id,lo,hi]))]);
    if(revision!==mdRevision)return;el('mdStatus').textContent=indices.length.toLocaleString()+' / '+data.count.toLocaleString()+' centers · z '+lo.toFixed(1)+'–'+hi.toFixed(1)+' Å';

  }catch(error){el('mdStatus').textContent=error.message;el('mdStatus').classList.add('error');throw error;}
}
let sampleRevision=0,sampleIdentity='';const samplePlotKeys=new Map();
async function drawSamples(){
  const revision=++sampleRevision,p=pair(),snapshot=el('mdSnapshot').value,family=families[p.descriptor.field];
  const nnAsset=D.samples[snapshot].neural[p.nn.id],descriptorAsset=D.samples[snapshot].descriptors[family];
  await Promise.all([loadAsset(nnAsset.asset),loadAsset(descriptorAsset.asset)]);if(revision!==sampleRevision)return;
  const nnData=window.CLUSTER_SAMPLES[nnAsset.key],descriptorData=window.CLUSTER_SAMPLES[descriptorAsset.key];
  let latticeNN={},latticeDescriptors={};
  if(D.lattice&&['sampleStructure','sampleGrid','sampleResidual'].some(id=>el(id).checked)){const a=D.lattice[snapshot].neural[p.nn.id],b=D.lattice[snapshot].descriptors[family];await Promise.all([loadAsset(a.asset),loadAsset(b.asset)]);if(revision!==sampleRevision)return;latticeNN=window.LATTICE_SAMPLES[a.key];latticeDescriptors=window.LATTICE_SAMPLES[b.key];}
  const left=nnData.clusters,right=descriptorData.clusters,identity=snapshot+'/'+p.nn.id+'/'+family+'/'+el('color').value;
  if(identity!==sampleIdentity){
    for(const [control,clusters,neural] of [['sampleNeuralCluster',left,true],['sampleDescriptorCluster',right,false]]){
      const select=el(control),old=select.value;select.replaceChildren();
      const order=Array.from({length:7},(_,i)=>i).sort((a,b)=>neural?p.map[a]-p.map[b]:a-b);
      for(const id of order)select.add(new Option(label(id,neural,p.map)+' · '+clusters[id].count.toLocaleString()+' centers',id));
      if(old!=='')select.value=old;
    }
    sampleIdentity=identity;
  }
  const selected=[];
  for(const [prefix,clusters,neural] of [['sampleNeural',left,true],['sampleDescriptor',right,false]]){
    const cluster=Number(el(prefix+'Cluster').value),members=clusters[cluster],select=el(prefix+'Index'),old=Number(select.value||0);
    select.replaceChildren();members.rows.forEach((row,i)=>select.add(new Option(String(i+1),i)));
    select.disabled=members.rows.length===0;select.value=Math.min(old,members.rows.length-1);
    const row=members.rows[Number(select.value)],patch=members.rows.length?(neural?nnData:descriptorData).patches[row]:null;
    el(prefix+'Info').textContent=patch?'Center atom '+patch.atom+' · example '+(Number(select.value)+1)+' / '+members.rows.length:'No members in this snapshot';
    const fit=patch?(neural?latticeNN:latticeDescriptors)[row]:null;
    if(fit&&(el('sampleStructure').checked||el('sampleGrid').checked||el('sampleResidual').checked))el(prefix+'Info').textContent+=' · '+(fit.accepted?fit.candidate:'No crystal identified; best candidate '+fit.candidate)+(fit.rmsd===null?'':' · PTM RMSD '+fit.rmsd.toFixed(3)+' · full-sample mismatch '+fit.extended_rms_A.toFixed(2)+' Å');
    selected.push({prefix,cluster,neural,patch,fit});
  }
  const limit=Math.max(1,...selected.flatMap(s=>s.patch?[...s.patch.xyz,...(((el('sampleGrid').checked||el('sampleResidual').checked)&&s.fit)?s.fit.matched.map(i=>s.fit.grid[i]):[])].flat().map(Math.abs):[]))*1.08;
  await Promise.all(selected.map(s=>{
    const xyz=s.patch?s.patch.xyz:[],colorIndex=s.neural&&el('color').value!=='original'?p.map[s.cluster]:s.cluster,traces=[];
    if(s.patch){
      const edgeXYZ=[[],[],[]],edgeColors=[];
      s.patch.edges.forEach(([a,b],i)=>{for(let axis=0;axis<3;axis++)edgeXYZ[axis].push(xyz[a][axis],xyz[b][axis],null);edgeColors.push(...Array(3).fill(palette[colorIndex]));});
      traces.push({type:'scatter3d',mode:'lines',x:edgeXYZ[0],y:edgeXYZ[1],z:edgeXYZ[2],line:{color:edgeColors,width:2.7},opacity:.8,hoverinfo:'skip',showlegend:false},
        {type:'scatter3d',mode:'markers',x:xyz.map(v=>v[0]),y:xyz.map(v=>v[1]),z:xyz.map(v=>v[2]),marker:{size:xyz.map((_,i)=>i===0?11:9),color:palette[colorIndex],line:{color:'#222222',width:.8}},hoverinfo:'skip',showlegend:false});
    }
    if(s.fit&&s.patch)traces.push(...latticeOverlays(s.patch,s.fit,palette[colorIndex]));
    const axis=()=>({visible:false,range:[-limit,limit]});
    const key=JSON.stringify([snapshot,s.neural?p.nn.id:family,s.cluster,s.patch?.atom,colorIndex,limit,['sampleStructure','sampleGrid','sampleResidual'].map(id=>el(id).checked)]);
    if(samplePlotKeys.get(s.prefix)===key)return;
    samplePlotKeys.set(s.prefix,key);
    return Plotly.react(s.prefix,traces,{...common,margin:{t:0,b:0,l:0,r:0},scene:{aspectmode:'cube',uirevision:s.prefix,xaxis:axis(),yaxis:axis(),zaxis:axis(),camera:{projection:{type:'orthographic'},eye:{x:1.315,y:1.027,z:.674}}}},config);
  }));
}
if(D.samples){
  el('environmentSamples').hidden=false;
  el('sampleNeuralCluster').addEventListener('change',()=>{el('sampleDescriptorCluster').value=pair().map[Number(el('sampleNeuralCluster').value)];el('sampleNeuralIndex').value=0;el('sampleDescriptorIndex').value=0;requestViewer(['samples']);});
  ['sampleDescriptorCluster','sampleNeuralIndex','sampleDescriptorIndex','sampleStructure','sampleGrid','sampleResidual'].forEach(id=>el(id).addEventListener('change',()=>requestViewer(['samples'])));
}
['nnSpace','descriptorSpace','color'].forEach(id=>el(id).addEventListener('change',()=>{frameRender=requestViewer();}));
el('frame').addEventListener('change',()=>{frameRender=requestViewer(['projection','correspondence']);});
el('highlightInterface').addEventListener('change',()=>{frameRender=requestViewer(['projection']);});
el('size').addEventListener('input',()=>{frameRender=requestViewer(['projection']);});
el('matrixScale').addEventListener('change',()=>requestViewer(['correspondence']));
el('mdSnapshot').addEventListener('change',()=>{frameRender=requestViewer(['md','samples','travel']);});
['mdSize','zLow','zHigh'].forEach(id=>el(id).addEventListener('input',()=>{frameRender=requestViewer(['md']);}));
function collapsePanels(){document.querySelectorAll('.panel.expanded').forEach(panel=>{panel.classList.remove('expanded');panel.querySelector('.expand').textContent='Expand';Plotly.Plots.resize(panel.querySelector('.plot'));});}
document.querySelectorAll('.expand').forEach(button=>button.addEventListener('click',()=>{
  const panel=button.closest('.panel'),wasExpanded=panel.classList.contains('expanded');collapsePanels();
  if(!wasExpanded){panel.classList.add('expanded');button.textContent='Close';Plotly.Plots.resize(el(button.dataset.plot));}
}));
document.addEventListener('keydown',event=>{if(event.key==='Escape')collapsePanels();});
// Only visible sections load assets and render. Dirty off-screen sections consume
// the latest selection on entry; intermediate slider states never form a queue.
const viewerSections=new Map();
function sectionVisible(section){return section.nodes.some(node=>{const r=node.getBoundingClientRect();return r.height>0&&r.bottom>=-100&&r.top<=innerHeight+100;});}
async function runSection(section){
 if(section.running||!section.dirty||!sectionVisible(section))return section.running;
 section.running=(async()=>{
  while(section.dirty&&sectionVisible(section)){
   section.dirty=false;await section.draw();
  }
 })().catch(error=>{section.dirty=true;console.error(error);throw error;}).finally(()=>{section.running=null;});
 return section.running;
}
function requestViewer(names=[...viewerSections.keys()]){
 for(const name of names){const section=viewerSections.get(name);if(!section)continue;section.dirty=true;
  if(name==='correspondence')++correspondenceRevision;if(name==='projection')++projectionRevision;if(name==='md')++mdRevision;if(name==='samples')++sampleRevision;if(name==='travel')++travelRevision;
 }
 return flushViewer();
}
async function flushViewer(){
 await new Promise(resolve=>requestAnimationFrame(resolve));
 await Promise.all([...viewerSections.values()].map(runSection));
}
let correspondenceKey='',correspondenceRevision=0;
async function renderCorrespondence(){
 const revision=correspondenceRevision,p=pair(),indices=selectedRows(),key=JSON.stringify([p.nn.id,p.descriptor.id,rowSelectionKey,el('matrixScale').value]);
 if(key===correspondenceKey)return;
 await loadAsset(p.nn.asset);if(revision!==correspondenceRevision)return;if(D.explorer)D.fields[p.nn.field]=window.PACMAP_LAYOUTS[p.nn.key].clusters;
 await drawCorrespondence(indices,p);correspondenceKey=key;
}
for(const [name,ids,draw] of [
 ['projection',['two','three'],drawProjection],['md',['mdNN','mdFeatures'],drawMD],
 ['samples',['environmentSamples'],drawSamples],['correspondence',['correspondenceSection'],renderCorrespondence],
 ['travel',['embeddingTravel'],prepareTravel]]){
 if(name==='samples'&&!D.samples||name==='travel'&&!D.travel)continue;
 if(name==='travel')el('embeddingTravel').hidden=false;
 viewerSections.set(name,{nodes:ids.map(el),draw,dirty:true,running:null});
}
const visibilityObserver=new IntersectionObserver(entries=>{
 for(const entry of entries)if(entry.isIntersecting){const section=[...viewerSections.values()].find(s=>s.nodes.includes(entry.target));if(section)runSection(section);}
},{rootMargin:'100px'});
for(const section of viewerSections.values())for(const node of section.nodes)visibilityObserver.observe(node);
frameRender=flushViewer();
