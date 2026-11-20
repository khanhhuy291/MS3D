import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

const $ = id => document.getElementById(id);
const colors = {1: '#41cbe8', 2: '#ffc76a', 3: '#f18ebc', 4: '#a5b4fc', 5: '#f97316'};
const names = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Barrier', 5: 'Traffic cone'};
const sourceNames = {lyft: 'Lyft', nuscenes: 'nuScenes', bevfusion: 'BEVFusion', centerpoint: 'CenterPoint', voxelrcnn: 'VoxelRCNN'};
const format = n => Number(n).toLocaleString('vi-VN');
const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {
  catalog: null, frame: null, index: 0, caseId: 'fusion3', rawPoints: null,
  points: new Float32Array(), pointRGB: new Float32Array(), pointPacked: new Uint32Array(),
  primary: [], comparison: [], selected: null, loading: false, playing: false,
  playTimer: null, request: null, generation: 0, pointOriginal: 0,
  bev: {x: 0, y: 0, range: 40}, webgl: false, three: null, ready: false,
};

function caseInfo(id = state.caseId) { return state.catalog.cases.find(c => c.id === id); }
function selectedBox() {
  return [...state.primary, ...state.comparison].find(d => d.id === state.selected?.id && d.caseId === state.selected?.caseId);
}
function filtered(caseId) {
  if (!state.frame || !state.frame.cases[caseId]) return [];
  const score = +$('score').value;
  const classes = new Set([...document.querySelectorAll('.class-check:checked')].map(x => +x.value));
  const kind = $('groupFilter').value, source = $('sourceFilter').value;
  return state.frame.cases[caseId].filter(d => d.score + 1e-9 >= score && classes.has(d.class_id)
    && (kind === 'all' || d.kind === kind) && (source === 'all' || d.sources.includes(source)))
    .map(d => ({...d, caseId}));
}
function updateFilters() {
  if (!state.frame) return;
  $('scoreValue').textContent = (+$('score').value).toFixed(2);
  state.primary = filtered(state.caseId);
  state.comparison = $('compareEnabled').checked && $('compareCase').value !== state.caseId ? filtered($('compareCase').value) : [];
  if (state.selected && !selectedBox()) state.selected = null;
  $('activeTitle').textContent = caseInfo().title;
  $('visibleCount').textContent = `${state.primary.length} box`;
  $('caseDescription').textContent = caseInfo().description;
  $('compareCase').disabled = !$('compareEnabled').checked;
  for (const button of document.querySelectorAll('.case-option')) {
    button.classList.toggle('active', button.dataset.case === state.caseId);
    button.querySelector('.count').textContent = filtered(button.dataset.case).length;
  }
  const notice = [];
  if (+$('score').value < caseInfo().stored_min_score) notice.push(`Kết quả này đã được lọc từ score ${caseInfo().stored_min_score}. Giảm thanh score không khôi phục các box đã bị loại.`);
  if (state.caseId === 'bevfusion') notice.push('PCD gốc có intensity bằng 0. BEVFusion đang dùng một sweep; xem kết quả với giới hạn này.');
  if (!caseInfo().has_provenance && ($('sourceFilter').value !== 'all' || !['all', 'unknown'].includes($('groupFilter').value))) notice.push('Kết quả này không lưu thành viên nguồn từng box; dùng bộ lọc “Tất cả” hoặc “Nguồn chưa được lưu”.');
  $('notice').hidden = !notice.length; $('notice').textContent = notice.join(' ');
  updateTable(); updateStats(); updateSelection(); updateThreeBoxes(); drawBEV();
  history.replaceState(null, '', `?frame=${state.index}&case=${encodeURIComponent(state.caseId)}&score=${$('score').value}`);
}

function setCase(id) {
  if (!caseInfo(id)) return;
  state.caseId = id; state.selected = null; updateFilters();
}

function updateTable() {
  $('tableCount').textContent = `(${state.primary.length})`;
  $('boxTable').innerHTML = state.primary.length ? [...state.primary].sort((a,b) => b.score-a.score).map(d => {
    const b = d.box;
    const sources = d.sources.length ? d.sources.map(s => sourceNames[s]).join(' + ') : 'Chưa lưu';
    return `<tr data-id="${escape(d.id)}"><td>#${d.index}</td><td><i class="dot" style="background:${colors[d.class_id]}"></i> ${names[d.class_id]}</td><td>${d.score.toFixed(3)}</td><td>${sources}</td><td>${b.slice(0,3).map(v=>v.toFixed(2)).join(' / ')}</td><td>${b.slice(3,6).map(v=>v.toFixed(2)).join(' × ')}</td></tr>`;
  }).join('') : '<tr><td class="empty-row" colspan="6">Không có box thỏa bộ lọc.</td></tr>';
}

function updateStats() {
  $('classStats').innerHTML = state.catalog.classes.map(({id:c}) => `<div class="class-stat"><i class="dot" style="background:${colors[c]}"></i>${names[c]}<strong>${state.primary.filter(d => d.class_id === c).length}</strong></div>`).join('');
  $('groupCount').textContent = ['single','pair','triple'].map(k => state.primary.filter(d => d.kind === k).length).join(' / ');
  $('pointCount').textContent = `${format(state.points.length/3)} / ${format(state.pointOriginal)}`;
}

function updateSelection() {
  const d = selectedBox();
  for (const row of $('boxTable').querySelectorAll('tr')) row.classList.toggle('selected', row.dataset.id === d?.id && d?.caseId === state.caseId);
  if (!d) {
    $('selectionDetails').className = 'selection-empty';
    $('selectionDetails').textContent = 'Bấm vào box trong BEV, 3D hoặc danh sách để xem chi tiết.';
    return;
  }
  $('selectionDetails').className = '';
  const b = d.box, labels = {single:'1 nguồn · singleton',pair:'2 nguồn',triple:'3 nguồn',unknown:'Không lưu nguồn'};
  $('selectionDetails').innerHTML = `<div class="selection-title"><i class="dot" style="background:${colors[d.class_id]}"></i> ${names[d.class_id]} <span class="subtle">#${d.index}</span></div>
    <div class="hint">${escape(caseInfo(d.caseId).title)}</div>
    ${d.original_name ? `<div class="selection-row"><span>Lớp gốc</span><strong>${escape(d.original_name)}</strong></div>` : ''}
    <div class="selection-row"><span>Score</span><strong>${d.score.toFixed(4)}</strong></div>
    <div class="selection-row"><span>Nhóm</span><strong>${labels[d.kind]}</strong></div>
    <div>${d.sources.map(s=>`<span class="source-tag">${sourceNames[s]} #${d.source_indices[s]}</span>`).join('')}</div>
    <div class="selection-row"><span>X / Y / Z</span><strong>${b.slice(0,3).map(v=>v.toFixed(2)).join(' / ')}</strong></div>
    <div class="selection-row"><span>DX × DY × DZ</span><strong>${b.slice(3,6).map(v=>v.toFixed(2)).join(' × ')}</strong></div>
    <div class="selection-row"><span>Yaw · trục DX</span><strong>${(b[6]*180/Math.PI).toFixed(1)}°</strong></div>
    <button class="small-button" id="focusSelected" style="margin-top:8px;width:100%">Đưa box vào giữa góc 3D</button>`;
  $('focusSelected').onclick = () => focusBox(d);
}
function selectDetection(d, focus = false) {
  state.selected = d ? {id:d.id, caseId:d.caseId} : null;
  updateSelection(); updateThreeBoxes(); drawBEV();
  if (focus && d) focusBox(d);
}

async function loadFrame(index) {
  index = Math.min(state.catalog.frames.length-1, Math.max(0, Math.round(index)));
  state.request?.abort(); const controller = new AbortController(); state.request = controller;
  const generation = ++state.generation;
  state.loading = true;
  for (const id of ['bevWrap','threeWrap']) $(id).classList.add('loading');
  $('loadStatus').textContent = `Đang nạp frame ${index}…`;
  try {
    const started = performance.now();
    const options = {signal:controller.signal};
    const [frameResponse, pointsResponse] = await Promise.all([
      fetch(`/api/frame/${index}`, options),
      fetch(`/api/points/${index}?max_points=${$('pointBudget').value}`, options),
    ]);
    if (!frameResponse.ok || !pointsResponse.ok) throw new Error('Không đọc được frame hoặc point cloud. Xem log server.');
    const [frame, buffer] = await Promise.all([frameResponse.json(), pointsResponse.arrayBuffer()]);
    if (generation !== state.generation) return;
    if (buffer.byteLength % 12) throw new Error('Point cloud binary không đúng định dạng XYZ.');
    state.index = index; state.frame = frame; state.rawPoints = new Float32Array(buffer);
    state.pointOriginal = +pointsResponse.headers.get('X-Original-Point-Count');
    state.selected = null;
    $('timeline').value = index; $('frameInput').value = index;
    $('frameName').textContent = frame.frame_id;
    $('frameTimestamp').textContent = `+${frame.elapsed_seconds.toFixed(3)} s · ${frame.timestamp.toFixed(6)}`;
    $('previous').disabled = index === 0; $('next').disabled = index === state.catalog.frames.length-1;
    $('fatalError').hidden = true;
    updatePoints(); updateFilters(); state.ready = true;
    $('loadStatus').textContent = `${frame.frame_id} · ${format(frame.num_points)} điểm gốc · nạp ${(performance.now()-started).toFixed(0)} ms`;
  } catch (error) {
    if (error.name !== 'AbortError') {
      $('fatalError').hidden = false; $('fatalError').textContent = error.message;
      $('loadStatus').textContent = 'Lỗi nạp dữ liệu'; stopPlayback();
    }
  } finally {
    if (generation === state.generation) {
      state.loading = false;
      for (const id of ['bevWrap','threeWrap']) $(id).classList.remove('loading');
    }
  }
}

function updatePoints() {
  if (!state.rawPoints) return;
  let low = +$('zMin').value, high = +$('zMax').value;
  if (!Number.isFinite(low) || !Number.isFinite(high) || low > high) return;
  const original = state.rawPoints, accepted = [];
  for (let i=0;i<original.length;i+=3) if (original[i+2]>=low && original[i+2]<=high) accepted.push(original[i],original[i+1],original[i+2]);
  state.points = new Float32Array(accepted);
  state.pointRGB = new Float32Array(accepted.length);
  state.pointPacked = new Uint32Array(accepted.length/3);
  for (let i=0;i<accepted.length;i+=3) {
    const t = Math.max(0,Math.min(1,(accepted[i+2]+.3)/3.5));
    const r = .32+.45*t, g=.53+.05*t, b=.65-.32*t;
    state.pointRGB.set([r,g,b],i);
    state.pointPacked[i/3] = (255<<24)|((b*255|0)<<16)|((g*255|0)<<8)|(r*255|0);
  }
  if (state.webgl) {
    const t = state.three;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.BufferAttribute(state.points,3));
    geometry.setAttribute('color',new THREE.BufferAttribute(state.pointRGB,3));
    t.cloud.geometry.dispose(); t.cloud.geometry = geometry;
    t.cloud.visible = $('showPoints').checked;
    t.cloud.material.size = +$('pointSize').value;
    t.dirty = true;
  }
  $('pointSizeValue').textContent = (+$('pointSize').value).toFixed(1);
  updateStats(); drawBEV();
}

const bevCanvas = $('bevCanvas');
function bevMetrics() {
  const width = bevCanvas.clientWidth, height = bevCanvas.clientHeight;
  return {width,height,scale:Math.min(width,height)/(2*state.bev.range)};
}
function projectBEV(x,y) {
  const {width,height,scale}=bevMetrics();
  return [width/2-(y-state.bev.y)*scale,height/2-(x-state.bev.x)*scale];
}
function worldBEV(u,v) {
  const {width,height,scale}=bevMetrics();
  return [state.bev.x-(v-height/2)/scale,state.bev.y-(u-width/2)/scale];
}
function rectangle(d) {
  const [x,y,z,dx,dy,dz,yaw] = d.box, c=Math.cos(yaw), s=Math.sin(yaw);
  return [[dx/2,dy/2],[dx/2,-dy/2],[-dx/2,-dy/2],[-dx/2,dy/2]].map(([a,b])=>projectBEV(x+a*c-b*s,y+a*s+b*c));
}
function drawBEV() {
  const {width,height,scale}=bevMetrics(); if(!width || !height) return;
  const dpr=Math.min(window.devicePixelRatio||1,2), W=Math.round(width*dpr), H=Math.round(height*dpr);
  if(bevCanvas.width!==W || bevCanvas.height!==H) {bevCanvas.width=W;bevCanvas.height=H;}
  const ctx=bevCanvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  const image=ctx.createImageData(W,H), pixels=new Uint32Array(image.data.buffer); pixels.fill(0xff1c1209);
  if($('showPoints').checked) {
    const pointSize=Math.max(1,Math.round(+$('pointSize').value*dpr));
    for(let i=0;i<state.points.length;i+=3) {
      const u=Math.round((width/2-(state.points[i+1]-state.bev.y)*scale)*dpr);
      const v=Math.round((height/2-(state.points[i]-state.bev.x)*scale)*dpr);
      if(u<0||v<0||u>=W||v>=H) continue;
      for(let j=0;j<pointSize && v+j<H;j++) for(let k=0;k<pointSize && u+k<W;k++) pixels[(v+j)*W+u+k]=state.pointPacked[i/3];
    }
  }
  ctx.putImageData(image,0,0);
  if($('showGrid').checked) {
    ctx.save(); ctx.globalAlpha=.45; ctx.strokeStyle='#31475b';ctx.lineWidth=.6;ctx.setLineDash([]);
    const [xmin,ymin]=worldBEV(width,height),[xmax,ymax]=worldBEV(0,0);
    for(let x=Math.floor(xmin/10)*10;x<=xmax;x+=10){const a=projectBEV(x,ymin),b=projectBEV(x,ymax);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
    for(let y=Math.floor(ymin/10)*10;y<=ymax;y+=10){const a=projectBEV(xmin,y),b=projectBEV(xmax,y);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
    ctx.restore();
  }
  for(const [list,comparison] of [[state.comparison,true],[state.primary,false]]) for(const d of list) {
    const selected=d.id===state.selected?.id && d.caseId===state.selected?.caseId;
    const corners=rectangle(d); ctx.beginPath();ctx.moveTo(...corners[0]);for(const p of corners.slice(1))ctx.lineTo(...p);ctx.closePath();
    ctx.strokeStyle=selected?'#ffffff':comparison?'#b7a1ff':colors[d.class_id];
    ctx.lineWidth=selected?2.5:comparison?1:1.3;ctx.globalAlpha=comparison ? .62 : 1;
    ctx.setLineDash(comparison?[2,4]:d.kind==='single'?[5,3]:[]);ctx.stroke();
    if(selected){ctx.fillStyle='#ffffff15';ctx.fill();}
    const [x,y,z,dx,dy,dz,yaw]=d.box, center=projectBEV(x,y),head=projectBEV(x+dx/2*Math.cos(yaw),y+dx/2*Math.sin(yaw));
    ctx.setLineDash([]);ctx.beginPath();ctx.moveTo(...center);ctx.lineTo(...head);ctx.stroke();
    if(!comparison && ($('showScores').checked || selected)){ctx.font='10px ui-monospace,monospace';ctx.fillStyle=selected?'#ffffff':colors[d.class_id];ctx.fillText(d.score.toFixed(2),center[0]+4,center[1]-4);}
  }
  ctx.globalAlpha=1;ctx.setLineDash([]);
  const origin=projectBEV(0,0);ctx.strokeStyle='#68dec6';ctx.lineWidth=1.6;ctx.beginPath();ctx.moveTo(origin[0]-5,origin[1]);ctx.lineTo(origin[0]+5,origin[1]);ctx.moveTo(origin[0],origin[1]-5);ctx.lineTo(origin[0],origin[1]+5);ctx.stroke();
  ctx.fillStyle='#829ab0';ctx.font='9px ui-monospace,monospace';ctx.fillText('EGO',origin[0]+8,origin[1]+4);
  const tick=10*scale;ctx.strokeStyle='#7895ae';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(15,height-32);ctx.lineTo(15+tick,height-32);ctx.stroke();ctx.fillText('10 m',15,height-38);
  $('bevScale').textContent=`±${state.bev.range.toFixed(0)} m · vehicle`;
}

function initThree() {
  try {
    const wrap=$('threeWrap');
    const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));renderer.setClearColor('#09121c');wrap.appendChild(renderer.domElement);
    const scene=new THREE.Scene();
    const camera=new THREE.PerspectiveCamera(48,1,.1,1000);camera.up.set(0,0,1);
    const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.12;controls.minDistance=2;controls.maxDistance=250;
    const grid=new THREE.GridHelper(160,16,0x40586b,0x25394c);grid.rotation.x=Math.PI/2;grid.position.z=-.015;grid.material.transparent=true;grid.material.opacity=.45;scene.add(grid);
    const axes=new THREE.AxesHelper(5);scene.add(axes);
    const ego=new THREE.Mesh(new THREE.SphereGeometry(.18,12,8),new THREE.MeshBasicMaterial({color:0x68dec6}));scene.add(ego);
    const cloud=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({size:1.5,vertexColors:true,sizeAttenuation:false}));cloud.frustumCulled=false;scene.add(cloud);
    const boxes=new THREE.Group();scene.add(boxes);
    const cube=new THREE.BoxGeometry(1,1,1),edges=new THREE.EdgesGeometry(cube);
    const materials={};
    for(const [id,color] of Object.entries(colors)) materials[id]={
      mesh:new THREE.MeshBasicMaterial({color,transparent:true,opacity:.035,depthWrite:false}),
      line:new THREE.LineBasicMaterial({color}),
      dashed:new THREE.LineDashedMaterial({color,dashSize:.12,gapSize:.07}),
    };
    materials.compare={mesh:new THREE.MeshBasicMaterial({color:0xb7a1ff,transparent:true,opacity:.012,depthWrite:false}),line:new THREE.LineDashedMaterial({color:0xb7a1ff,transparent:true,opacity:.6,dashSize:.07,gapSize:.06})};
    materials.selected={mesh:new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:.1,depthWrite:false}),line:new THREE.LineBasicMaterial({color:0xffffff})};
    state.three={renderer,scene,camera,controls,grid,cloud,boxes,cube,edges,materials,dirty:true};state.webgl=true;
    controls.addEventListener('change',()=>{state.three.dirty=true;});
    renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();$('renderStatus').textContent='WebGL context lost';});
    let down=null;
    renderer.domElement.addEventListener('pointerdown',event=>{down=[event.clientX,event.clientY];});
    renderer.domElement.addEventListener('pointerup',event=>{
      if(!down || Math.hypot(event.clientX-down[0],event.clientY-down[1])>5) return;
      const rect=renderer.domElement.getBoundingClientRect(),mouse=new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);
      const ray=new THREE.Raycaster();ray.params.Line.threshold=.12;ray.setFromCamera(mouse,camera);
      const hits=ray.intersectObjects(boxes.children,true);const hit=hits.find(x=>x.object.userData.detection);
      selectDetection(hit?.object.userData.detection||null);
    });
    renderer.domElement.addEventListener('dblclick',()=>{const d=selectedBox();if(d)focusBox(d);});
    setCamera('iso'); resize();
    function loop(){requestAnimationFrame(loop);controls.update();if(state.three.dirty){renderer.render(scene,camera);state.three.dirty=false;}}
    loop();$('renderStatus').textContent=renderer.capabilities.isWebGL2?'WebGL 2':'WebGL';
  } catch(error) {
    state.webgl=false;$('renderStatus').textContent='Không có WebGL';$('threeError').hidden=false;
    $('threeError').textContent=`Không tạo được WebGL: ${error.message}. BEV vẫn dùng được. Hãy bật hardware acceleration trong trình duyệt để xem 3D.`;
  }
}

function updateThreeBoxes() {
  if(!state.webgl) return;
  const t=state.three;
  for(const group of [...t.boxes.children]) {t.boxes.remove(group);for(const child of group.children)if(child.userData.ownGeometry)child.geometry.dispose();}
  for(const [list,comparison] of [[state.comparison,true],[state.primary,false]]) for(const d of list) {
    const [x,y,z,dx,dy,dz,yaw]=d.box;
    const group=new THREE.Group();group.position.set(x,y,z);group.rotation.z=yaw;group.scale.set(dx,dy,dz);
    const selected=d.id===state.selected?.id && d.caseId===state.selected?.caseId;
    const material=selected?t.materials.selected:comparison?t.materials.compare:t.materials[d.class_id];
    const mesh=new THREE.Mesh(t.cube,material.mesh);mesh.userData.detection=d;group.add(mesh);
    const geometry=t.edges.clone();
    const outline=new THREE.LineSegments(geometry,material.line || material.dashed);
    if(!comparison && !selected && d.kind==='single') outline.material=material.dashed;
    outline.computeLineDistances();outline.userData.detection=d;outline.userData.ownGeometry=true;group.add(outline);
    t.boxes.add(group);
  }
  t.grid.visible=$('showGrid').checked;t.dirty=true;
}
function setCamera(preset) {
  if(!state.webgl) return;
  const t=state.three;t.camera.up.set(0,0,1);t.controls.target.set(0,0,0);
  if(preset==='top')t.camera.position.set(0,-.001,85);
  else if(preset==='front')t.camera.position.set(70,0,4);
  else if(preset==='left')t.camera.position.set(0,70,5);
  else t.camera.position.set(42,-48,42);
  t.camera.lookAt(t.controls.target);t.controls.update();t.dirty=true;
  for(const button of document.querySelectorAll('[data-camera]'))button.classList.toggle('active',button.dataset.camera===preset);
}
function focusBox(d) {
  if(!state.webgl)return;
  const t=state.three,target=new THREE.Vector3(...d.box.slice(0,3));
  const offset=t.camera.position.clone().sub(t.controls.target).normalize().multiplyScalar(Math.max(10,d.box[3]*3));
  t.controls.target.copy(target);t.camera.position.copy(target).add(offset);t.controls.update();t.dirty=true;
}
function resize() {
  drawBEV();
  if(state.webgl){const t=state.three,w=$('threeWrap').clientWidth,h=$('threeWrap').clientHeight;t.renderer.setSize(w,h);t.camera.aspect=w/h;t.camera.updateProjectionMatrix();t.dirty=true;}
}

function pickBEV(u,v) {
  const [x,y]=worldBEV(u,v);let best=null,distance=Infinity;
  for(const d of [...state.primary,...state.comparison]) {
    const [cx,cy,z,dx,dy,dz,yaw]=d.box,rx=x-cx,ry=y-cy;
    const localX=rx*Math.cos(yaw)+ry*Math.sin(yaw),localY=-rx*Math.sin(yaw)+ry*Math.cos(yaw);
    if(Math.abs(localX)<=dx/2+.15 && Math.abs(localY)<=dy/2+.15){const dist=Math.hypot(rx,ry);if(dist<distance){best=d;distance=dist;}}
  }
  selectDetection(best);
}
function zoomBEV(factor, u=null,v=null) {
  const before=u===null?null:worldBEV(u,v);state.bev.range=Math.max(4,Math.min(150,state.bev.range*factor));
  if(before){const after=worldBEV(u,v);state.bev.x+=before[0]-after[0];state.bev.y+=before[1]-after[1];}
  drawBEV();
}
function stopPlayback(){state.playing=false;clearTimeout(state.playTimer);$('play').textContent='▶';$('play').setAttribute('aria-label','Phát');}
function startPlayback(){
  state.playing=true;$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','Dừng');
  async function tick(){if(!state.playing)return;await loadFrame((state.index+1)%state.catalog.frames.length);if(state.playing)state.playTimer=setTimeout(tick,1000/+$('playbackSpeed').value);}
  state.playTimer=setTimeout(tick,1000/+$('playbackSpeed').value);
}
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);}
function screenshot(){
  if(!state.frame)return;
  if(state.webgl)state.three.renderer.render(state.three.scene,state.three.camera);
  const canvases=[bevCanvas];if(state.webgl)canvases.push(state.three.renderer.domElement);
  const output=document.createElement('canvas');output.width=canvases.reduce((s,c)=>s+c.width,0);output.height=Math.max(...canvases.map(c=>c.height))+75;
  const ctx=output.getContext('2d');ctx.fillStyle='#0b1017';ctx.fillRect(0,0,output.width,output.height);ctx.fillStyle='#e5edf6';ctx.font='18px sans-serif';ctx.fillText(`MS3D · ${state.frame.frame_id} · ${caseInfo().title}`,20,26);ctx.font='12px sans-serif';ctx.fillStyle='#9bb1c5';ctx.fillText(`Vehicle frame · score >= ${$('score').value} · ${state.primary.length} boxes${state.comparison.length?' · đối chiếu '+caseInfo($('compareCase').value).title:''}`,20,48);
  let x=0;for(const c of canvases){ctx.drawImage(c,x,65);x+=c.width;}
  output.toBlob(blob=>{if(blob)download(blob,`${state.frame.frame_id}_${state.caseId}_bev_3d.png`);});
}

function bindEvents(){
  for(const id of ['score','groupFilter','sourceFilter','compareEnabled','compareCase','showScores'])$(id).addEventListener('input',updateFilters);
  document.querySelectorAll('.class-check').forEach(x=>x.addEventListener('change',updateFilters));
  $('resetFilters').onclick=()=>{$('score').value=.3;$('groupFilter').value='all';$('sourceFilter').value='all';document.querySelectorAll('.class-check').forEach(x=>x.checked=true);updateFilters();};
  $('previous').onclick=()=>{stopPlayback();loadFrame(state.index-1);};$('next').onclick=()=>{stopPlayback();loadFrame(state.index+1);};
  $('play').onclick=()=>state.playing?stopPlayback():startPlayback();
  let sliderTimer; $('timeline').addEventListener('input',()=>{stopPlayback();$('frameInput').value=$('timeline').value;clearTimeout(sliderTimer);sliderTimer=setTimeout(()=>loadFrame(+$('timeline').value),90);});
  $('frameInput').addEventListener('change',()=>{stopPlayback();loadFrame(+$('frameInput').value);});
  $('pointBudget').onchange=()=>loadFrame(state.index);
  for(const id of ['pointSize','zMin','zMax'])$(id).addEventListener('input',updatePoints);
  $('showPoints').onchange=()=>{if(state.webgl){state.three.cloud.visible=$('showPoints').checked;state.three.dirty=true;}drawBEV();};
  $('showGrid').onchange=()=>{updateThreeBoxes();drawBEV();};
  $('clearSelection').onclick=()=>selectDetection(null);
  $('boxTable').onclick=event=>{const tr=event.target.closest('tr[data-id]');if(tr)selectDetection(state.primary.find(d=>d.id===tr.dataset.id));};
  $('boxTable').ondblclick=event=>{const tr=event.target.closest('tr[data-id]');if(tr)selectDetection(state.primary.find(d=>d.id===tr.dataset.id),true);};
  $('bevZoomIn').onclick=()=>zoomBEV(.8);$('bevZoomOut').onclick=()=>zoomBEV(1.25);$('bevReset').onclick=()=>{state.bev={x:0,y:0,range:40};drawBEV();};
  document.querySelectorAll('[data-camera]').forEach(b=>b.onclick=()=>setCamera(b.dataset.camera));
  let drag=null;
  bevCanvas.oncontextmenu=e=>e.preventDefault();
  bevCanvas.addEventListener('pointerdown',event=>{bevCanvas.setPointerCapture(event.pointerId);drag={x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,moved:false};});
  bevCanvas.addEventListener('pointermove',event=>{if(!drag)return;const {scale}=bevMetrics();if(Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)>4)drag.moved=true;if(drag.moved){state.bev.x+=(event.clientY-drag.y)/scale;state.bev.y+=(event.clientX-drag.x)/scale;drawBEV();}drag.x=event.clientX;drag.y=event.clientY;});
  bevCanvas.addEventListener('pointerup',event=>{if(drag && !drag.moved){const r=bevCanvas.getBoundingClientRect();pickBEV(event.clientX-r.left,event.clientY-r.top);}drag=null;});
  bevCanvas.addEventListener('dblclick',()=>{const d=selectedBox();if(d)focusBox(d);});
  bevCanvas.addEventListener('wheel',event=>{event.preventDefault();const rect=bevCanvas.getBoundingClientRect();zoomBEV(event.deltaY>0?1.1:1/1.1,event.clientX-rect.left,event.clientY-rect.top);},{passive:false});
  $('screenshot').onclick=screenshot;
  $('exportJson').onclick=()=>{if(state.frame)download(new Blob([JSON.stringify({frame_id:state.frame.frame_id,timestamp:state.frame.timestamp,coordinate_frame:'vehicle',units:'metres',case_id:state.caseId,score_min:+$('score').value,filters:{class_ids:[...document.querySelectorAll('.class-check:checked')].map(x=>+x.value),group:$('groupFilter').value,contributing_source:$('sourceFilter').value},detections:state.primary,comparison_case_id:$('compareEnabled').checked?$('compareCase').value:null,comparison_detections:state.comparison},null,2)],{type:'application/json'}),`${state.frame.frame_id}_${state.caseId}.json`);};
  window.addEventListener('resize',resize);
  new ResizeObserver(resize).observe($('bevWrap'));
  document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName))return;if(event.code==='Space'){event.preventDefault();$('play').click();}if(event.code==='ArrowRight')$('next').click();if(event.code==='ArrowLeft')$('previous').click();if(event.code==='KeyR'){setCamera('iso');$('bevReset').click();}});
}

async function init(){
  const response=await fetch('/api/catalog');if(!response.ok)throw new Error('Không đọc được catalog.');
  state.catalog=await response.json();state.caseId=state.catalog.default_case;
  document.querySelector('.class-filters').innerHTML=state.catalog.classes.map(c=>`<label><input type="checkbox" class="class-check" value="${c.id}" checked><i class="dot" style="background:${c.color}"></i>${escape(c.name)}</label>`).join('');
  const params=new URLSearchParams(location.search);if(caseInfo(params.get('case')))state.caseId=params.get('case');
  if(params.has('score'))$('score').value=Math.max(.1,Math.min(.99,+params.get('score')||.3));
  $('frameCountBadge').textContent=`${state.catalog.frames.length} frames`;$('frameMax').textContent=`/ ${state.catalog.frames.length-1}`;
  $('timeline').max=state.catalog.frames.length-1;$('frameInput').max=state.catalog.frames.length-1;$('caseTotal').textContent=`${state.catalog.cases.length} phiên bản`;
  $('caseList').innerHTML=state.catalog.cases.map(c=>`<button class="case-option" data-case="${c.id}"><span class="case-radio"></span><span>${escape(c.title)}</span><span class="count">—</span></button>`).join('');
  $('caseList').querySelectorAll('button').forEach(b=>b.onclick=()=>setCase(b.dataset.case));
  $('compareCase').innerHTML=state.catalog.cases.map(c=>`<option value="${c.id}">${escape(c.title)}</option>`).join('');$('compareCase').value='fusion2';
  initThree();bindEvents();await loadFrame(+params.get('frame')||0);
  window.__MS3D_VIEWER__={snapshot:()=>({ready:state.ready,loading:state.loading,index:state.index,caseId:state.caseId,visibleBoxes:state.primary.length,comparisonBoxes:state.comparison.length,displayedPoints:state.points.length/3,webgl:state.webgl,selected:state.selected,camera:state.three?.camera.position.toArray()})};
}
init().catch(error=>{$('fatalError').hidden=false;$('fatalError').textContent=error.message;$('loadStatus').textContent='Không khởi tạo được viewer';console.error(error);});
