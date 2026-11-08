/* Dependency-free 3D scene: WebGL cortex, canvas tract/electrode glow. */
const $ = id => document.getElementById(id);
const brainCanvas = $('brain'), overlay = $('overlay'), ctx = overlay.getContext('2d');
const gl = brainCanvas.getContext('webgl', {alpha:true, antialias:true});
const state = {scene:null, yaw:-0.52, pitch:0.17, zoom:1, playing:false, time:0,
  min:0, max:1, selected:null, lastFrame:0, center:[0,0,0], extent:1};

function shader(type, source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}
let program, positionBuffer, normalBuffer, vertexCount=0;
if(gl){
  program=gl.createProgram();
  gl.attachShader(program,shader(gl.VERTEX_SHADER,`attribute vec3 aPos;attribute vec3 aNormal;uniform mat3 uRot;uniform vec3 uCenter;uniform float uExtent;uniform float uZoom;uniform float uAspect;varying vec3 vNormal;varying vec3 vView;void main(){vec3 p=uRot*((aPos-uCenter)/uExtent)*uZoom;gl_Position=vec4(p.x/uAspect,p.y,p.z*.45,1.);vNormal=normalize(uRot*aNormal);vView=p;}`));
  gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 vNormal;varying vec3 vView;void main(){vec3 n=normalize(vNormal);float key=max(dot(n,normalize(vec3(-.35,.7,.65))),0.);float fill=max(dot(n,normalize(vec3(.75,-.1,.25))),0.);float rim=pow(1.-abs(n.z),2.2);vec3 deep=vec3(.18,.33,.38);vec3 pearl=vec3(.58,.76,.77);vec3 color=mix(deep,pearl,.3+key*.48+fill*.1)+vec3(.12,.34,.36)*rim;gl_FragColor=vec4(color,.37+rim*.1);}`));
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
  positionBuffer=gl.createBuffer();normalBuffer=gl.createBuffer();
}
function rot(){let y=state.yaw,p=state.pitch,cy=Math.cos(y),sy=Math.sin(y),cp=Math.cos(p),sp=Math.sin(p);
  return [cy,sy*sp,sy*cp,0,cp,-sp,-sy,cy*sp,cy*cp]}
function turn(v){let m=rot();return [m[0]*v[0]+m[3]*v[1]+m[6]*v[2],m[1]*v[0]+m[4]*v[1]+m[7]*v[2],m[2]*v[0]+m[5]*v[1]+m[8]*v[2]]}
function project(point){let v=turn(point.map((x,i)=>(x-state.center[i])/state.extent));let unit=Math.min(overlay.width,overlay.height)*.47*state.zoom;return [overlay.width*.5+v[0]*unit,overlay.height*.51-v[1]*unit,v[2]]}
function buildBrain(){if(!gl)return;let vertices=[],normals=[];
  for(const surface of state.scene.surfaces){const v=surface.vertices;for(const f of surface.faces){const a=v[f[0]],b=v[f[1]],c=v[f[2]];
    let u=b.map((x,i)=>x-a[i]),w=c.map((x,i)=>x-a[i]);let n=[u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]];let len=Math.hypot(...n)||1;n=n.map(x=>x/len);
    for(const x of [a,b,c]){vertices.push(...x);normals.push(...n)}
  }}vertexCount=vertices.length/3;
  gl.bindBuffer(gl.ARRAY_BUFFER,positionBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(vertices),gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER,normalBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(normals),gl.STATIC_DRAW);
}
function measureScene(){let xyz=[];for(const e of state.scene.electrodes)xyz.push(e.xyz);for(const s of state.scene.surfaces){for(let i=0;i<s.vertices.length;i+=20)xyz.push(s.vertices[i])}for(const t of state.scene.tracts){if(t.length)xyz.push(t[0],t[t.length-1])}
  if(!xyz.length)return;let low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];for(const v of xyz)for(let i=0;i<3;i++){low[i]=Math.min(low[i],v[i]);high[i]=Math.max(high[i],v[i])}
  state.center=low.map((x,i)=>(x+high[i])/2);state.extent=Math.max(...high.map((x,i)=>x-low[i]))*.59||1;
}
function resize(){let dpr=Math.min(devicePixelRatio||1,2);for(const c of [brainCanvas,overlay]){c.width=Math.round(c.clientWidth*dpr);c.height=Math.round(c.clientHeight*dpr)}if(gl)gl.viewport(0,0,brainCanvas.width,brainCanvas.height)}
function renderBrain(){if(!gl)return;gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);if(!vertexCount||!$('show-brain').checked)return;
  gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.disable(gl.CULL_FACE);gl.useProgram(program);
  const m=rot();gl.uniformMatrix3fv(gl.getUniformLocation(program,'uRot'),false,new Float32Array(m));gl.uniform3fv(gl.getUniformLocation(program,'uCenter'),new Float32Array(state.center));gl.uniform1f(gl.getUniformLocation(program,'uExtent'),state.extent);gl.uniform1f(gl.getUniformLocation(program,'uZoom'),state.zoom*.94);gl.uniform1f(gl.getUniformLocation(program,'uAspect'),brainCanvas.width/brainCanvas.height);
  gl.bindBuffer(gl.ARRAY_BUFFER,positionBuffer);let p=gl.getAttribLocation(program,'aPos');gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,normalBuffer);let n=gl.getAttribLocation(program,'aNormal');gl.enableVertexAttribArray(n);gl.vertexAttribPointer(n,3,gl.FLOAT,false,0,0);gl.drawArrays(gl.TRIANGLES,0,vertexCount)
}
function path(points,color,width,glow){if(!points||points.length<2)return;let projected=points.map(project);ctx.beginPath();ctx.moveTo(projected[0][0],projected[0][1]);for(const p of projected.slice(1))ctx.lineTo(p[0],p[1]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.lineJoin='round';ctx.shadowColor=color;ctx.shadowBlur=glow;ctx.stroke();ctx.shadowBlur=0}
function dot(x,y,r,color,glow){ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=color;ctx.shadowColor=color;ctx.shadowBlur=glow;ctx.fill();ctx.shadowBlur=0}
function drawOverlay(){ctx.clearRect(0,0,overlay.width,overlay.height);const scene=state.scene;if(!scene)return;const scale=Math.min(overlay.width,overlay.height)/700;
  if($('show-tracts').checked){let stride=Math.max(1,Math.ceil(scene.tracts.length/340));for(let i=0;i<scene.tracts.length;i+=stride)path(scene.tracts[i],'rgba(80,210,208,.15)',.7*scale,0)}
  for(const lead of scene.depth_electrodes||[]){
    path(lead.path,'rgba(199,225,231,.62)',2.1*scale,8*scale);
    const [x,y]=project(lead.path[0]);ctx.font=`600 ${Math.max(10,12*scale)}px DM Sans,Arial`;
    ctx.fillStyle='#b6d5db';ctx.shadowColor='#83dcd8';ctx.shadowBlur=7*scale;
    ctx.fillText(lead.name,x+8*scale,y-6*scale);ctx.shadowBlur=0;
  }
  let candidates=scene.propagations.filter(r=>r.start_s-.08<=state.time&&r.end_s+.13>=state.time);if(state.selected!=null){let selected=scene.propagations[state.selected];if(selected&&!candidates.includes(selected))candidates.unshift(selected)}
  for(const r of candidates.slice(0,12)){let color=r.kind==='Spike'?'#ff5ea8':r.kind==='HFO'?'#ffbd70':'#96a5ff';path(r.path,color,3*scale,16*scale);
    let fraction=Math.max(0,Math.min(1,(state.time-r.start_s)/(r.end_s-r.start_s||.001)));let index=Math.min(r.path.length-1,Math.floor(fraction*(r.path.length-1)));let pt=project(r.path[index]);dot(pt[0],pt[1],5*scale,color,24*scale)}
  let active=new Map();for(const ev of scene.events){if(Math.abs(ev.peak_s-state.time)<.09){if(ev.kind==='Spike'&&!$('show-spike').checked)continue;if(ev.kind==='HFO'&&!$('show-hfo').checked)continue;active.set(ev.channel,ev.kind)}}
  let sorted=[...scene.electrodes].sort((a,b)=>project(a.xyz)[2]-project(b.xyz)[2]);for(const e of sorted){let [x,y]=project(e.xyz),kind=active.get(e.channel),color=kind==='Spike'?'#ff5ea8':kind==='HFO'?'#ffbd70':'#bcecea';
    const radius=e.contact_diameter_mm?Math.max(1.4,(e.contact_diameter_mm/2)*Math.min(overlay.width,overlay.height)*.47*state.zoom/state.extent):3.5*scale;
    dot(x,y,radius,color,kind?30*scale:7*scale);
    if(kind){ctx.beginPath();ctx.arc(x,y,radius+4*scale,0,Math.PI*2);ctx.strokeStyle=color;ctx.lineWidth=1.3*scale;ctx.stroke();ctx.font=`${Math.max(10,11*scale)}px DM Sans,Arial`;ctx.fillStyle='#e7f7f7';ctx.fillText(e.label,x+11*scale,y-10*scale)}}
}
function formatTime(seconds){let m=Math.floor(seconds/60),s=Math.floor(seconds%60),ms=Math.floor((seconds%1)*1000);return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(ms).padStart(3,'0')}`}
function updateUi(){const scene=state.scene;if(!scene)return;$('current-time').textContent=formatTime(state.time);$('timeline').value=state.time;$('play').textContent=state.playing?'Ⅱ':'▶';let cards=document.querySelectorAll('.candidate');cards.forEach((c,i)=>c.classList.toggle('active',i===state.selected))}
function frame(ts){if(state.playing&&state.lastFrame){state.time+=Math.min((ts-state.lastFrame)/1000,.1)*Number($('speed').value);
    if(state.scene?.demo){const next=state.scene.events.find(e=>e.peak_s>state.time+.35);const soon=state.scene.events.some(e=>e.peak_s>=state.time&&e.peak_s<=state.time+.35);if(!soon&&next)state.time=next.peak_s-.16}
    if(state.time>=state.max){state.time=state.min;state.playing=false}}state.lastFrame=ts;renderBrain();drawOverlay();updateUi();requestAnimationFrame(frame)}
function setupUi(){let scene=state.scene;const times=scene.events.map(e=>e.peak_s);let first=times.length?Math.min(...times):0,last=times.length?Math.max(...times):1;state.min=scene.demo?Math.max(0,first-.18):Math.min(first,0);state.max=scene.demo?last+.2:Math.max(last,1);if(state.max-state.min<.2)state.max=state.min+1;state.time=state.min;
  $('source-label').textContent=scene.source_label;$('event-count').textContent=scene.events.length;$('route-count').textContent=scene.propagations.length;$('electrode-count').textContent=scene.electrodes.length;$('lead-count').textContent=(scene.depth_electrodes||[]).length||'—';
  if(scene.demo)$('speed').value='0.1';
  $('window-label').textContent=`${scene.settings.min_delay_ms}–${scene.settings.max_delay_ms} MS`;$('timeline').min=state.min;$('timeline').max=state.max;$('range-label').textContent=`${formatTime(state.min)}  —  ${formatTime(state.max)}`;
  $('ticks').innerHTML=scene.events.slice(0,500).map(e=>`<i class="tick ${e.kind==='HFO'?'hfo':''}" style="left:${100*(e.peak_s-state.min)/(state.max-state.min)}%"></i>`).join('');
  const eById=new Map(scene.events.map(e=>[e.id,e]));let container=$('candidates');if(!scene.propagations.length){container.innerHTML='<div class="empty">No pathway candidates in the current time and tract settings.</div>'}else{
    scene.propagations.slice(0,80).forEach((r,i)=>{let a=scene.electrodes.find(e=>e.channel_number===r.source_channel),b=scene.electrodes.find(e=>e.channel_number===r.target_channel);let card=document.createElement('button');card.className='candidate';
      const top=document.createElement('div');top.className='candidate-top';let kind=document.createElement('b');kind.className=r.kind;kind.textContent=r.kind.toUpperCase();let delay=document.createElement('span');delay.textContent=`${r.delay_ms} ms`;top.append(kind,delay);
      let route=document.createElement('div');route.className='candidate-route';route.textContent=a?.label||String(r.source_channel);let arrow=document.createElement('span');arrow.textContent='→';route.append(arrow,document.createTextNode(b?.label||String(r.target_channel)));
      let bottom=document.createElement('div');bottom.className='candidate-bottom';bottom.textContent=`${formatTime(r.start_s)}  ·  ${r.length_mm} mm tract segment`;
      card.append(top,route,bottom);card.onclick=()=>{state.selected=i;state.time=r.start_s;state.playing=false};container.append(card)})}
  $('play').onclick=()=>{if(state.time>=state.max)state.time=state.min;state.playing=!state.playing};$('timeline').oninput=e=>{state.time=Number(e.target.value);state.playing=false};$('reset-view').onclick=()=>{state.yaw=-.52;state.pitch=.17;state.zoom=1};
}
let dragging=false,lastX=0,lastY=0;overlay.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;overlay.setPointerCapture(e.pointerId)});overlay.addEventListener('pointermove',e=>{if(!dragging)return;state.yaw+=(e.clientX-lastX)*.008;state.pitch=Math.max(-1.35,Math.min(1.35,state.pitch+(e.clientY-lastY)*.008));lastX=e.clientX;lastY=e.clientY});overlay.addEventListener('pointerup',()=>dragging=false);overlay.addEventListener('wheel',e=>{e.preventDefault();state.zoom=Math.max(.55,Math.min(2.8,state.zoom*Math.exp(-e.deltaY*.001)))},{passive:false});
window.addEventListener('resize',resize);
fetch('/scene.json').then(r=>{if(!r.ok)throw Error(`Scene request failed: ${r.status}`);return r.json()}).then(scene=>{state.scene=scene;measureScene();buildBrain();setupUi();resize();requestAnimationFrame(frame)}).catch(err=>{$('candidates').textContent=err.message;console.error(err)});



