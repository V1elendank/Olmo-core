"use strict";
const $ = id => document.getElementById(id);
let session = null, assets = [], starting = false, removeId = null, requestId = null;
const active = {model: false, dataset: false};
const paused = {model: false, dataset: false};
const size = n => n >= 1024**3 ? (n/1024**3).toFixed(2)+' GB' : n >= 1024**2 ? (n/1024**2).toFixed(1)+' MB' : n+' bytes';
function element(tag, text, cls) { const el = document.createElement(tag); if (text != null) el.textContent=text; if(cls) el.className=cls; return el; }
function message(id, text, error=false) { $(id).textContent=text; $(id).classList.toggle('error',error); }
async function api(path, options={}) {
  const response=await fetch(path,{...options,headers:{...(options.method ? {'X-FT-Token':session?.token||''}:{}),...options.headers}});
  const data=await response.json();
  if (!response.ok) { const error=new Error(data.error || 'Request failed');error.status=response.status;throw error; }
  return data;
}
const post = (path, body) => api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function filesFor(kind) {
  return Array.from($(kind+'-files').files).map(file => {
    const path=kind==='model' ? (file.webkitRelativePath||file.name).split('/').slice(file.webkitRelativePath ? 1 : 0).join('/') : file.name;
    return {file,path,size:file.size,modified:file.lastModified};
  }).filter(f => !f.path.split('/').some(p=>p.startsWith('.')) && (kind!=='model'||/\.(json|safetensors|model|txt|jinja)$/.test(f.path))).sort((a,b)=>a.path.localeCompare(b.path));
}
function refreshStart() {
  const ready = session?.slurm_available && $('model-select').value && $('dataset-select').value;
  $('start').disabled = !ready || starting;
  $('start-help').textContent=starting ? 'Submitting once. You can follow progress below.' : !session?.slurm_available ? 'Start is available when this panel runs on the CS cluster.' : ready ? 'The main run starts only after the fit-check succeeds.' : 'Complete both uploads to enable Start.';
}
function architectureOptions() {
  const model=assets.find(a=>a.id===$('model-select').value);
  const current=$('arch').value;
  const options=(session?.architectures||[]).filter(a=>!model||a.startsWith(model.model.family+'_'));
  $('arch').replaceChildren(...options.map(a=>new Option(a,a)));
  const preferred=model ? model.model.family+'_7b' : 'olmo3_7b';
  $('arch').value=options.includes(current)&&current.startsWith(model?.model.family||'olmo3') ? current : options.includes(preferred) ? preferred : options[0]||'';
}
function renderAssets() {
  const list=$('assets'); list.replaceChildren();
  if(!assets.length) list.append(element('p','No uploads yet.','small'));
  for(const asset of assets) {
    const row=element('div',null,'asset'), info=element('div',null,'info');
    info.append(element('strong',asset.name),element('p',`${asset.kind} · ${size(asset.bytes)} · ${asset.state} · ${asset.id.slice(0,8)}`,'small'));
    if(asset.error) info.append(element('p',asset.error,'error'));
    if(asset.state==='uploading') info.append(element('p','To resume, select the same files above and click Upload again.','small'));
    const remove=element('button','Remove','secondary compact');
    remove.type='button'; remove.disabled=asset.state==='checking'||active[asset.kind];
    remove.onclick=()=>{removeId=asset.id;$('remove-text').textContent=asset.name;$('remove-dialog').showModal();};
    row.append(info,element('span',asset.state,'tag'),remove);list.append(row);
  }
  for(const kind of ['model','dataset']) {
    const select=$(kind+'-select'), previous=select.value;
    const ready=assets.filter(a=>a.kind===kind&&a.state==='ready');
    select.replaceChildren(new Option(`Choose ${kind} upload…`,''),...ready.map(a=>new Option(`${a.name} · ${a.id.slice(0,8)}`,a.id)));
    if(ready.some(a=>a.id===previous)) select.value=previous;
    else if(ready.length===1) select.value=ready[0].id;
  }
  architectureOptions();refreshStart();
}
async function refreshAssets() { assets=(await api('/api/assets')).assets;renderAssets(); }
async function upload(kind) {
  if(active[kind]||!session) return;
  const files=filesFor(kind), name=$(kind+'-name').value.trim();
  if(!name||!files.length) {message(kind+'-status','Choose files and give this upload a name.',true);return;}
  active[kind]=true;paused[kind]=false;
  $(kind+'-upload').disabled=true;$(kind+'-files').disabled=true;$(kind+'-pause').hidden=false;$(kind+'-progress').hidden=false;
  const key='ft-upload-'+kind;
  const signature=JSON.stringify(files.map(({path,size,modified})=>({path,size,modified})));
  try {
    let saved; try {saved=JSON.parse(localStorage.getItem(key)||'null');}catch(e){}
    let asset;
    if(saved?.signature===signature) {
      // A failed lookup must not silently start a second multi-GB upload.
      try {asset=(await api('/api/uploads/'+saved.id)).asset;}catch(e){if(e.status!==404)throw e;}
    }
    if(!asset) asset=(await post('/api/uploads',{kind,name,files:files.map(({path,size})=>({path,size}))})).asset;
    localStorage.setItem(key,JSON.stringify({id:asset.id,signature}));
    while(asset.state==='checking') {
      message(kind+'-status','The server is still checking this upload. Waiting for the result…');
      if(paused[kind]) throw new Error('Stopped waiting. Select the same files and click Upload to check again.');
      await new Promise(resolve=>setTimeout(resolve,1500));
      asset=(await api('/api/uploads/'+asset.id)).asset;
    }
    if(asset.state==='ready') {
      localStorage.removeItem(key);$(kind+'-progress').value=100;
      message(kind+'-status','Ready. Your previous upload completed successfully.');
      await refreshAssets();$(kind+'-select').value=asset.id;architectureOptions();refreshStart();return;
    }
    const offsets=(await api('/api/uploads/'+asset.id)).files;
    const total=files.reduce((sum,f)=>sum+f.size,0);
    let sent=offsets.reduce((sum,f)=>sum+f.offset,0);
    for(let index=0;index<files.length;index++) {
      const file=files[index];let offset=offsets[index].offset;
      while(offset<file.size) {
        if(paused[kind]) throw new Error('Upload paused. Select the same files and click Upload to resume.');
        const blob=file.file.slice(offset,Math.min(offset+session.chunk_bytes,file.size));
        const result=await api(`/api/uploads/${asset.id}/files/${index}`,{method:'PUT',headers:{'Content-Type':'application/octet-stream','X-Upload-Offset':String(offset)},body:blob});
        sent+=result.offset-offset;offset=result.offset;
        $(kind+'-progress').value=100*sent/total;
        message(kind+'-status',`${size(sent)} / ${size(total)} · ${file.path}`);
      }
    }
    $(kind+'-pause').hidden=true;
    message(kind+'-status','Upload complete. Checking files and calculating content hashes…');
    await post('/api/uploads/'+asset.id+'/complete',{});
    localStorage.removeItem(key);
    $(kind+'-progress').value=100;message(kind+'-status','Ready. Select this upload below to fine-tune.');
    await refreshAssets();$(kind+'-select').value=asset.id;architectureOptions();refreshStart();
  } catch(e) {message(kind+'-status',e.message,true);}
  finally {active[kind]=false;$(kind+'-files').disabled=false;$(kind+'-upload').disabled=!session;$(kind+'-pause').hidden=true;await refreshAssets().catch(()=>{});}
}
function renderRuns(runs) {
  const list=$('launches');list.replaceChildren();
  if(!runs.length) list.append(element('p','Your submitted runs will appear here.','small'));
  for(const run of runs) {
    const card=element('div',null,'launch');card.append(element('strong',run.name),element('span',' '+run.state,'tag'));
    if(run.error) card.append(element('p',run.error,'error'));
    if(run.state==='submitted') card.append(element('p','Jobs submitted to Slurm. Follow training completion and metrics in the dashboard.'));
    if(run.jobs?.length) card.append(element('p',run.jobs.map(j=>`${j.stage}: ${j.job_id}`).join(' · ')));
    card.append(element('p','Output directory: '+run.run_dir,'small'));
    if(run.log) {const details=element('details');details.append(element('summary','Submission log'),element('pre',run.log));card.append(details);}
    list.append(card);
  }
}
async function refreshRuns(){renderRuns((await api('/api/runs')).runs);}
$('run-form').addEventListener('submit',async e=>{
  e.preventDefault();if(starting||!session?.slurm_available)return;
  starting=true;refreshStart();message('run-error','');
  requestId=requestId||crypto.randomUUID().replaceAll('-','');
  const body={request_id:requestId,name:$('run-name').value.trim(),model_id:$('model-select').value,dataset_id:$('dataset-select').value,
    arch:$('arch').value,format:$('format').value,chat_template:$('chat-template').value,steps:Number($('steps').value),seq_len:Number($('seq-len').value),
    global_batch_seqs:Number($('batch').value),lr:Number($('lr').value),constraint:$('constraint').value,gpus:Number($('gpus').value),hours:Number($('hours').value)};
  for(const key of ['messages','prompt','response','text']) body[key+'_column']=$(key+'-column').value.trim();
  try {const result=await post('/api/runs',body);message('run-error',result.run.state==='failed' ? result.run.error : 'Launch accepted. Follow job IDs and any errors below.',result.run.state==='failed');await refreshRuns();}
  catch(e){message('run-error',e.message,true);}
  finally{starting=false;refreshStart();}
});
$('run-form').addEventListener('change',()=>{requestId=null;refreshStart();});
$('model-select').addEventListener('change',architectureOptions);
$('refresh-assets').onclick=()=>refreshAssets().catch(e=>message('asset-error',e.message,true));
$('remove-cancel').onclick=()=>$('remove-dialog').close();
$('remove-confirm').onclick=async()=>{
  $('remove-confirm').disabled=true;
  try {await api('/api/uploads/'+removeId,{method:'DELETE'});message('asset-error','');await refreshAssets();}
  catch(e){message('asset-error',e.message,true);}
  finally{$('remove-confirm').disabled=false;$('remove-dialog').close();}
};
for(const kind of ['model','dataset']) {
  $(kind+'-files').onchange=()=>{const files=filesFor(kind);message(kind+'-files-info',`${files.length} supported files · ${size(files.reduce((s,f)=>s+f.size,0))}`);if(!$(kind+'-name').value&&files.length)$(kind+'-name').value=kind==='model'?$(kind+'-files').files[0].webkitRelativePath.split('/')[0]:files[0].path;};
  $(kind+'-upload').onclick=()=>upload(kind);$(kind+'-pause').onclick=()=>{paused[kind]=true;message(kind+'-status','Pausing after the current chunk…');};
}
(async()=>{
  try {
    session=await api('/api/session');
    message('connection',session.slurm_available?'Connected to your cluster panel. Upload files separately, then start when ready.':'Local preview: uploads work, but training submission is disabled because Slurm is unavailable.');
    for(const kind of ['model','dataset'])$(kind+'-upload').disabled=false;
    await Promise.all([refreshAssets(),refreshRuns()]);
    setInterval(()=>refreshRuns().catch(()=>{}),5000);
  }catch(e){message('connection','Could not connect. Start the live panel on the cluster and open it through the SSH tunnel. '+e.message,true);}
})();
