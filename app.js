/* ========= Utilities ========= */
const $ = (sel, el=document) => el.querySelector(sel);
const $$ = (sel, el=document) => Array.from(el.querySelectorAll(sel));
function el(tag, props={}){const e=document.createElement(tag);Object.assign(e, props);return e}

// Softmax (with temperature)
function softmax(arr, T=1){
  const scaled = arr.map(v => v / T);
  const m = Math.max(...scaled);
  const ex = scaled.map(v => Math.exp(v-m));
  const s = ex.reduce((a,b) => a+b, 0);
  return ex.map(v => v/s);
}
// Weighted random choice
function randChoice(probs){const r=Math.random();let acc=0;for(let i=0;i<probs.length;i++){acc+=probs[i];if(r<=acc) return i}return probs.length-1}
// Dot product
function dot(a,b){return a.reduce((sum, v, i) => sum + v*b[i], 0)}

// Reusable Bar Chart Drawer (absolute by default; optional normalize)
function drawBarRow(container, items, { label='Probability', normalize=false } = {}){
  container.innerHTML='';
  const wrap=el('div');
  const maxP = Math.max(0.0001, ...items.map(item => item.p));
  items.forEach(({word, p})=>{
    const row=el('div');
    row.style.display='grid';
    row.style.gridTemplateColumns='120px 1fr 80px';
    row.style.alignItems='center';
    row.style.gap='.5rem';
    row.style.marginBottom = '.25rem';
    const w=el('div',{textContent:word,className:'pill'});
    const bar=el('div');
    bar.style.height='16px';
    bar.style.border='2px solid var(--ink)';
    bar.style.borderRadius='10px';
    bar.style.position='relative';
    bar.style.background = '#eee';
    const fill=el('div');
    fill.style.height='100%';
    fill.style.width = normalize ? `${Math.round((p / maxP) * 100)}%` : `${Math.round(p * 100)}%`;
    fill.style.background='linear-gradient(90deg,var(--accent),var(--accent2))';
    fill.style.borderRadius='8px';
    fill.style.transition='width .3s';
    const labelText = (label==='Probability') ? `${(p*100).toFixed(1)}%` : p.toFixed(3);
    const labelEl=el('span',{textContent: labelText, className:'small mono'});
    bar.appendChild(fill);
    row.appendChild(w); row.appendChild(bar); row.appendChild(labelEl);
    wrap.appendChild(row);
  });
  container.appendChild(wrap);
}

// Status box helper
function showStatus(container, message, type='ok'){
  container.innerHTML='';
  const box = el('div', { textContent: message, className: type === 'ok' ? 'status-box status-ok' : 'status-box status-err' });
  container.appendChild(box);
}

// Reset all
$('#reset-all').addEventListener('click',()=>location.reload());

/* === Scene 1: Next-word prediction (Sonnets full, interpolation+Laplace fixed) === */
(function(){
  const sketch=$('#s1-sketch');
  const meta=$('#s1-meta');
  const src=$('#s1-source');

  // Project Gutenberg — The Sonnets (public domain)
  const SONNETS_URL = 'https://www.gutenberg.org/cache/epub/1041/pg1041.txt';

  // Fallback (tiny) if fetch fails — still public-domain lines
  const FALLBACK = `
Shall I compare thee to a summer's day?
When, in disgrace with fortune and men's eyes, I all alone beweep my outcast state,
Let me not to the marriage of true minds admit impediments. Love is not love
Which alters when it alteration finds, or bends with the remover to remove.`;

  // Tokenizer: keep letters, digits, apostrophes; lowercase
  const NONWORD = new RegExp("[^A-Za-z0-9' ]+","g");
  function tokenize(text){
    return text.toLowerCase().replace(NONWORD,' ').split(' ').filter(Boolean);
  }

  // n‑gram structures (up to 3‑token context → 4‑gram counts)
  let tokens=[], vocab=[], index=new Map(), uniCounts={}, uniTotal=0;
  const maxN=4; // context length = maxN-1
  const keyFor = (ctx)=> ctx.length + '|' + ctx.join(' ');

  function buildFromText(text){
    tokens = tokenize(text);
    vocab = Array.from(new Set(tokens));
    index = new Map();
    for(let i=0;i<tokens.length;i++){
      for(let L=1; L<maxN; L++){
        if(i-L<0) break;
        const ctx = tokens.slice(i-L,i);
        const key = keyFor(ctx);
        if(!index.has(key)) index.set(key, Object.create(null));
        const nxt = tokens[i];
        index.get(key)[nxt] = (index.get(key)[nxt]||0) + 1;
      }
    }
    uniCounts = tokens.reduce((m,t)=>{m[t]=(m[t]||0)+1;return m;}, {});
    uniTotal = tokens.length;
  }

  // Trim Gutenberg front/back matter to SONNETS section only (no regex literals to avoid escaping in patch)
  function extractSonnets(txt){
    let t = txt;
    const START = '*** START OF THE PROJECT GUTENBERG EBOOK';
    const END   = '*** END OF THE PROJECT GUTENBERG EBOOK';
    const sIdx = t.indexOf(START);
    if(sIdx>=0){ t = t.slice(sIdx + START.length); }
    const eIdx = t.indexOf(END);
    if(eIdx>=0){ t = t.slice(0, eIdx); }
    const loverIdx = t.indexOf("A LOVER'S COMPLAINT");
    if(loverIdx>0) t = t.slice(0, loverIdx);
    return t;
  }

  // Laplace (+1) smoothing always ON
  function distFromCounts(counts){
    const d = Object.assign({}, counts);
    let total = Object.values(d).reduce((a,b)=>a+b,0);
    for(const w of vocab){ d[w]=(d[w]||0)+1; }
    total += vocab.length;
    const out={}; Object.entries(d).forEach(([w,c])=>out[w]=c/total); return out;
  }

  function getDists(prompt){
    const words = tokenize(prompt);
    const levels = {L3:null,L2:null,L1:null};
    for(let L=maxN-1; L>=1; L--){
      const ctx = words.slice(-L);
      const key = keyFor(ctx);
      if(index.has(key)){
        const counts = index.get(key);
        const dist = distFromCounts(counts); // smoothing fixed ON
        const matches = Object.values(counts).reduce((a,b)=>a+b,0);
        if(L===3) levels.L3={ctx,dist,matches};
        if(L===2) levels.L2={ctx,dist,matches};
        if(L===1) levels.L1={ctx,dist,matches};
      }
    }
    if(!levels.L1){
      // unigram fallback (also smoothed)
      levels.L1 = { ctx:[], dist: distFromCounts(uniCounts), matches: uniTotal };
    }
    return levels;
  }

  // Interpolated backoff fixed ON — default lambdas
  function interpolate(d3,d2,d1){
    let l3=0.6, l2=0.3, l1=0.1;
    const present = (d3?l3:0) + (d2?l2:0) + (d1?l1:0);
    if(present>0){ l3=d3?l3/present:0; l2=d2?l2/present:0; l1=d1?l1/present:0; }
    const acc={};
    const words = new Set([ ...Object.keys(d3||{}), ...Object.keys(d2||{}), ...Object.keys(d1||{}) ]);
    words.forEach(w=>{ const p=(d3?.[w]||0)*l3 + (d2?.[w]||0)*l2 + (d1?.[w]||0)*l1; if(p>0) acc[w]=p; });
    const Z = Object.values(acc).reduce((a,b)=>a+b,0)||1; Object.keys(acc).forEach(w=>acc[w]/=Z); return acc;
  }

  function render(){
    const prompt = $('#s1-prompt').value.trim();
    const {L3,L2,L1} = getDists(prompt);
    const finalDist = interpolate(L3?.dist, L2?.dist, L1?.dist);
    const items = Object.entries(finalDist)
      .map(([w,p])=>({word:w,p}))
      .sort((a,b)=>b.p-a.p)
      .slice(0,10);
    drawBarRow(sketch, items, { normalize:false });
    if(meta) meta.textContent = `Interpolated (λ3=0.6, λ2=0.3, λ1=0.1) • Laplace(+1) • vocab=${vocab.length} tokens=${tokens.length} • matches L3=${L3?.matches||0}, L2=${L2?.matches||0}, L1=${L1?.matches||0}`;
  }

  async function loadSonnets(){
    if(meta) meta.textContent = 'Loading full sonnets (Project Gutenberg)…';
    try{
      const resp = await fetch(SONNETS_URL, {mode:'cors'});
      const txt  = await resp.text();
      const body = extractSonnets(txt);
      buildFromText(body);
      if(src) src.innerHTML = `Corpus: Shakespeare's <em>Sonnets</em> (154). Public domain. Source: <a href="${SONNETS_URL}" target="_blank" rel="noopener">Project Gutenberg #1041</a>`;
      render();
    }catch(err){
      console.error(err);
      if(meta) meta.textContent = 'Failed to fetch sonnets (CORS/offline). Using small built‑in fallback.';
      buildFromText(FALLBACK);
      if(src) src.textContent = 'Corpus: Fallback sample from Shakespeare (public domain). Source: embedded excerpt.';
      render();
    }
  }

  // Wire up UI
  $('#s1-suggest').addEventListener('click', render);
  $('#s1-append').addEventListener('click', ()=>{
    const promptEl = $('#s1-prompt');
    const {L3,L2,L1} = getDists(promptEl.value);
    const finalDist = interpolate(L3?.dist, L2?.dist, L1?.dist);
    const top = Object.entries(finalDist).sort((a,b)=>b[1]-a[1])[0];
    if(top){ promptEl.value = (promptEl.value + ' ' + top[0]).trim(); }
    render();
  });
  $('#s1-prompt').addEventListener('keydown', e=>{ if(e.key==='Enter') render(); });

  // Load full corpus immediately
  loadSonnets();
})();

/* === Scene 3–4: n-gram toy (Patched + smoothing) === */
(function(){
  const corpus = [
    ['i','like','pizza'],
    ['i','like','pasta'],
    ['you','like','pizza'],
    ['i','like','tea','very','much']
  ];
  const vocab = Array.from(new Set(corpus.flat()));
  const sketch=$('#s3-sketch');
  function computeNGrams() {
    const n = +$('#s3-n').value;
    const contextStr = $('#s3-context').value.trim().toLowerCase();
    const context = contextStr ? contextStr.split(/\s+/) : [];
    const contextLen = n - 1;
    const smoothing = $('#s3-smooth').checked;
    if (contextLen <= 0) { showStatus(sketch, `Error: n must be 2 or greater.`, 'err'); return; }
    if (context.length !== contextLen) { showStatus(sketch, `Error: Context must be ${contextLen} words long for n=${n}.`, 'err'); return; }

    // Initialize counts
    const counts = {};
    let total = 0;
    if (smoothing){ vocab.forEach(w => { counts[w] = 1; }); total += vocab.length; }

    corpus.forEach(tokens => {
      for (let i = 0; i <= tokens.length - n; i++) {
        const window = tokens.slice(i, i + n);
        const windowContext = window.slice(0, contextLen);
        if (windowContext.every((word, j) => word === context[j])) {
          const nextWord = window[contextLen];
          counts[nextWord] = (counts[nextWord] || 0) + 1;
          total++;
        }
      }
    });

    if (total === (smoothing ? vocab.length : 0)) {
      showStatus(sketch, `No counts found for context "${contextStr}" (Data Sparsity).${smoothing ? ' (Smoothing added uniform prior)' : ''}`, 'ok');
      return;
    }

    const items = Object.entries(counts)
      .map(([w, c]) => ({word: w, p: c/total}))
      .sort((a, b) => b.p - a.p)
      .slice(0, 8); // keep it compact
    drawBarRow(sketch, items, { normalize:false });
  }
  $('#s3-calc').addEventListener('click', computeNGrams);
  $('#s3-n').addEventListener('input', () => { $('#s3-nv').textContent = $('#s3-n').value; computeNGrams(); });
  computeNGrams();
})();

/* === Scene 5–6: embeddings mini-map (Patched: numeric legend) === */
(function(){
  const canvas=$('#s5-canvas');const ctx=canvas.getContext('2d');
  const words=[
    {w:'king',v:[0.9,0.8]},{w:'queen',v:[0.85,0.95]},{w:'man',v:[0.6,0.4]},
    {w:'woman',v:[0.62,0.58]},{w:'pizza',v:[0.15,0.2]},{w:'pasta',v:[0.2,0.18]}
  ];
  const W=canvas.width,H=canvas.height;const pad=40;
  function toXY([x,y]){return [pad+x*(W-2*pad), H-pad-y*(H-2*pad)];}
  function cosine(a,b){const d=dot(a,b);const na=Math.hypot(...a);const nb=Math.hypot(...b);return d/(na*nb)}
  let selected=-1;
  function draw(){
    ctx.clearRect(0,0,W,H);ctx.strokeStyle='#ddd';ctx.lineWidth=1;
    for(let i=0;i<5;i++){const x=pad+i*(W-2*pad)/4;ctx.beginPath();ctx.moveTo(x,pad);ctx.lineTo(x,H-pad);ctx.stroke()}
    for(let i=0;i<5;i++){const y=pad+i*(H-2*pad)/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke()}
    ctx.strokeStyle='var(--ink)';ctx.lineWidth=2;ctx.strokeRect(pad,pad,W-2*pad,H-2*pad);
    if(selected>=0){
      const v0=words[selected].v;
      const lines=[];
      words.forEach((p,i)=>{
        if(i===selected) return;
        const sim=cosine(v0,p.v); lines.push({w:p.w,s:sim});
        const [x0,y0]=toXY(v0);const [x1,y1]=toXY(p.v);
        ctx.strokeStyle='rgba(42,123,210,'+(Math.max(0,sim)*0.8)+')';ctx.lineWidth=4*Math.max(0,sim);
        ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();
      });
      $('#s5-legend').textContent = lines.sort((a,b)=>b.s-a.s).map(r=>`${r.w}:${r.s.toFixed(2)}`).join('  |  ');
    } else { $('#s5-legend').textContent='Click a word to list cosine similarities.'; }
    words.forEach((p,i)=>{
      const [x,y]=toXY(p.v);
      ctx.fillStyle=i===selected? '#2a7bd2':'#000';
      ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#000';ctx.font='bold 14px ui-sans-serif';ctx.fillText(p.w,x+8,y-8);
    });
    drawArrow(words[2].v, words[3].v, '#f08b2f'); // man→woman
    drawArrow(words[0].v, words[1].v, '#f08b2f'); // king→queen
  }
  function drawArrow(a,b,color){
    const [x0,y0]=toXY(a);const [x1,y1]=toXY(b);
    ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y1);ctx.stroke();
    const ang=Math.atan2(y1-y0,x1-x0);const len=9;
    ctx.beginPath();ctx.moveTo(x1,y1);
    ctx.lineTo(x1-len*Math.cos(ang-0.4), y1-len*Math.sin(ang-0.4));
    ctx.lineTo(x1-len*Math.cos(ang+0.4), y1-len*Math.sin(ang+0.4));
    ctx.closePath();ctx.fillStyle=color;ctx.fill();
  }
  canvas.addEventListener('click',(ev)=>{
    const r=canvas.getBoundingClientRect();const x=ev.clientX-r.left;const y=ev.clientY-r.top;
    let hit=-1;words.forEach((p,i)=>{const [px,py]=toXY(p.v);if(Math.hypot(px-x,py-y)<10) hit=i});
    selected=hit;draw();
  });
  $('#s5-reset').addEventListener('click',()=>{selected=-1;draw()});
  draw();
})();

/* === Scene 7: toy NN (Patched: biases + hidden activations + noise) === */
(function(){
  const sketch=$('#s7-sketch');
  const labels=['Vertical','Horizontal','Diagonal','None'];
  const model = {
    W_ih: [
      [ 2.0, -1.0,  1.0],
      [-1.0,  2.0, -1.0],
      [ 2.0, -1.0, -1.0],
      [-1.0,  2.0,  1.0]
    ],
    b_h: [0.2, 0.2, 0.1],
    W_ho: [
      [ 3.0, -1.0, -1.0, -1.0],
      [-1.0,  3.0, -1.0, -1.0],
      [-1.0, -1.0,  3.0, -1.0]
    ],
    b_o: [0,0,0,0]
  };
  const inputs = {
    Vertical:   [1,0,1,0],
    Horizontal: [1,1,0,0],
    Diagonal:   [1,0,0,1],
    None:       [0,0,0,0],
    All:        [1,1,1,1]
  };
  const relu = x => Math.max(0, x);
  function predict(input) {
    let hidden = [0,0,0];
    for (let h = 0; h < 3; h++) {
      hidden[h] = model.b_h[h];
      for (let i = 0; i < 4; i++) hidden[h] += input[i] * model.W_ih[i][h];
    }
    const hidden_activated = hidden.map(relu);
    let output = [0,0,0,0];
    for (let o = 0; o < 4; o++) {
      output[o] = model.b_o[o];
      for (let h = 0; h < 3; h++) output[o] += hidden_activated[h] * model.W_ho[h][o];
    }
    return { probs: softmax(output), hidden: hidden_activated };
  }
  function render(type){
    sketch.innerHTML = '';
    const input = inputs[type];
    const grid=el('div');grid.style.display='grid';grid.style.gridTemplateColumns='repeat(2,30px)';grid.style.gap='4px';grid.style.marginBottom='1rem';
    input.forEach(pixel=>{const d=el('div');d.style.width='28px';d.style.height='28px';d.style.border='2px solid var(--ink)';d.style.borderRadius='6px';d.style.background= pixel ? 'var(--accent)':'#fff';grid.appendChild(d)});
    sketch.appendChild(grid);
    // Hidden activations
    const { probs, hidden } = predict(input);
    const hItems = hidden.map((v,i)=>({word:`h${i+1}`, p: v/Math.max(1, Math.max(...hidden))}));
    drawBarRow(sketch, hItems, { label:'Activation', normalize:true });
    // Output probabilities
    const items = labels.map((w,i) => ({word:w, p:probs[i]})).sort((a,b) => b.p - a.p);
    drawBarRow(sketch, items, { normalize:false });
  }
  ['Vertical','Horizontal','Diagonal','None'].forEach(id => { $('#s7-'+id.toLowerCase()).addEventListener('click',()=>render(id)) });
  $('#s7-all').addEventListener('click',()=>render('All'));
  render('Vertical');
})();

/* === Scene 8: attention (Patched: show V*) === */
(function(){
  const sketch=$('#s8-sketch');
  const d_k = 2; // Key/Query dim
  const tokens = [
    { w: 'turn',   q: [1.0, 0.1], k: [1.0, 0.1], v: [0.2, 0.8] },
    { w: 'right',  q: [0.8, 0.5], k: [0.8, 0.5], v: [0.5, 0.5] },
    { w: 'at',     q: [0.1, 0.9], k: [0.1, 0.9], v: [0.1, 0.1] },
    { w: 'the',    q: [0.1, 0.9], k: [0.1, 0.9], v: [0.1, 0.1] },
    { w: 'light',  q: [0.9, 0.9], k: [0.9, 0.9], v: [0.7, 0.3] },
    { w: ',',      q: [0.0, 0.0], k: [0.0, 0.0], v: [0.0, 0.0] },
    { w: 'right?', q: [0.8, 0.4], k: [0.8, 0.4], v: [0.5, 0.6] }
  ];
  $('#s8-focus').innerHTML = tokens.map((t,i) => `<option value="${i}">${i}: ${t.w}</option>`).join('');
  function render(){
    const focusIdx = +$('#s8-focus').value;
    const q = tokens[focusIdx].q;
    const scores = tokens.map(t => dot(q, t.k) / Math.sqrt(d_k));
    const weights = softmax(scores);
    const items = tokens.map((t, i) => ({ word: `attn(Q_${tokens[focusIdx].w}→K_${t.w})`, p: weights[i] }))
                        .sort((a,b)=>b.p-a.p);
    sketch.innerHTML = '';
    const row=el('div');row.style.display='flex';row.style.gap='8px';row.style.flexWrap='wrap';
    tokens.forEach((t,i)=>{const b=el('div',{textContent:t.w,className:'pill'});if(i===focusIdx){b.style.background='var(--accent)';b.style.color='#fff';}row.appendChild(b)});
    sketch.appendChild(row);
    sketch.appendChild(el('hr', {style:'border:0; border-top: 1px dashed var(--muted); margin: 1rem 0;'}));
    drawBarRow(sketch, items, { label:'Weight', normalize:false });
    // V* display
    const vstar = tokens.reduce((acc,t,i)=>[acc[0]+weights[i]*t.v[0], acc[1]+weights[i]*t.v[1]], [0,0]);
    sketch.appendChild(el('div',{className:'small mono',textContent:`V* = [${vstar[0].toFixed(2)}, ${vstar[1].toFixed(2)}]`}));
  }
  $('#s8-focus').addEventListener('change', render);
  render();
})();

/* === Scene 9: training vs post-training === */
(function(){
  const sketch=$('#s9-sketch');let mode=0;
  function draw(){
    sketch.innerHTML='';
    const row=el('div');row.style.display='grid';row.style.gridTemplateColumns='1fr 60px 1fr';row.style.alignItems='center';row.style.gap='10px';
    const left=el('div',{className:'card',innerHTML:'<h3>Pretraining</h3><p class="small">Internet‑scale text, next‑token loss</p>'});
    const arrow=el('div',{innerHTML:'➡️',style:'text-align:center;font-size:28px'});
    const right=el('div',{className:'card',innerHTML:'<h3>Post‑training</h3><p class="small">SFT + preferences, refusals, tone</p>'});
    if(mode%2) left.style.filter='grayscale(1)', right.style.boxShadow='0 0 12px rgba(240,139,47,.6)';
    else right.style.filter='grayscale(1)', left.style.boxShadow='0 0 12px rgba(42,123,210,.6)';
    row.append(left,arrow,right);sketch.appendChild(row);
  }
  $('#s9-toggle').addEventListener('click',()=>{mode++;draw()});draw();
})();

/* === Scene 11: TIC builder === */
(function(){
  function buildPrompt(){
    const T=$('#s11-t').value.trim();const I=$('#s11-i').value.trim();const C=$('#s11-c').value.trim();
    const card=el('div');card.className='card';
    card.innerHTML=`<h3>Structured Prompt</h3>
      <div style="border-bottom: 2px dashed var(--muted); padding-bottom:.5rem; margin-bottom:.5rem;"><strong>TASK</strong><p>${T||'<span class="small">(not set)</span>'}</p></div>
      <div style="border-bottom: 2px dashed var(--muted); padding-bottom:.5rem; margin-bottom:.5rem;"><strong>INSTRUCTIONS</strong><p>${I||'<span class="small">(not set)</span>'}</p></div>
      <div><strong>CONTEXT</strong><p>${C||'<span class="small">(not set)</span>'}</p></div>`;
    const host=$('#s11-sketch');host.innerHTML='';host.appendChild(card);
  }
  $('#s11-build').addEventListener('click', buildPrompt);
  buildPrompt();
})();

/* === Scene 12: CoT === */
(function(){
  const host=$('#s12-sketch');
  function render(mode){
    host.innerHTML='';
    const wrap=el('div');wrap.style.display='grid';wrap.style.gridTemplateColumns='1fr 1fr';wrap.style.gap='10px';
    const fast=el('div',{className:'sketch',innerHTML:'<h3>Fast Answer</h3><p class="small">"The ball is $0.10" (Wrong)</p><div style="height:6px;background:var(--danger);width:80%;border-radius:6px;"></div>'});
    const cot=el('div',{className:'sketch',innerHTML:'<h3>Step‑by‑Step</h3><ol class="small" style="padding-left:1.2rem;margin:0;"><li>Let B=Bat, L=Ball</li><li>B + L = 1.10</li><li>B = L + 1.00</li><li>(L+1) + L = 1.10 → 2L = 0.10</li><li><strong>L = 0.05</strong> (Correct)</li></ol>'});
    if(mode==='cot') cot.style.boxShadow='0 0 15px rgba(42,123,210,.5)'; else fast.style.boxShadow='0 0 15px rgba(195,51,51,.5)';
    wrap.append(fast,cot);host.appendChild(wrap);
  }
  $('#s12-fast').addEventListener('click',()=>render('fast'));
  $('#s12-cot').addEventListener('click',()=>render('cot'));
  render('cot');
})();

/* === Scene 15: hidden stack === */
(function(){
  const host=$('#s15-sketch');
  function render(){
    host.innerHTML='';
    const active=$$('.s15').filter(x=>x.checked).map(x=>x.dataset.key);
    const stack=el('div');stack.style.position='relative';stack.style.height='220px';stack.style.padding='1rem';
    let top=160; active.forEach((name,i)=>{
      const card=el('div',{className:'card',textContent:name});
      card.style.position='absolute';card.style.left=(10+i*12)+'px';card.style.top=(top-i*28)+'px';card.style.width='calc(100% - 60px)';card.style.textAlign='center';card.style.background='rgba(255,254,249,.85)';
      stack.appendChild(card);
    });
    const prompt=el('div',{className:'pill',textContent:'Your Prompt'});prompt.style.position='absolute';prompt.style.left='50%';prompt.style.transform='translateX(-50%)';prompt.style.top='20px';prompt.style.background='var(--accent)';prompt.style.color='#fff';prompt.style.zIndex='10';
    stack.appendChild(prompt);host.appendChild(stack);
  }
  $$('.s15').forEach(cb=>cb.addEventListener('change',render));render();
})();

/* === Scene 16: agent loop === */
(function(){
  const host=$('#s16-sketch');let phase=0;const steps=['Plan','Act (Tool)','Read (Result)','Adjust'];
  function draw(){
    host.innerHTML='';
    const ring=el('div');ring.style.position='relative';ring.style.height='220px';ring.style.border='2px dashed var(--muted)';ring.style.borderRadius='50%';ring.style.background='#fff';
    steps.forEach((s,i)=>{const angle=i*Math.PI/2 - Math.PI/2; const x=100+80*Math.cos(angle);const y=100+80*Math.sin(angle); const chip=el('div',{className:'pill',textContent:s}); chip.style.position='absolute';chip.style.left=x+'px';chip.style.top=y+'px';chip.style.transform='translate(-50%,-50%)'; if(i===phase%4) chip.style.background='var(--accent2)', chip.style.color='#fff'; ring.appendChild(chip);});
    host.appendChild(ring);
  }
  $('#s16-step').addEventListener('click',()=>{phase++;draw()});draw();
})();

/* === Scene 17: temperature (Patched absolute bars + highlight) === */
(function(){
  const vocab=['clear','concise','creative','playful','formal'];
  const logits=[1.6,1.5,1.2,1.0,0.9];
  const host=$('#s17-sketch');
  function render(){
    const T=+$('#s17-T').value;$('#s17-Tv').textContent=T.toFixed(1);
    const probs=softmax(logits, T);
    drawBarRow(host, vocab.map((w,i)=>({word:w,p:probs[i]})), { normalize:false });
  }
  $('#s17-T').addEventListener('input',render);
  $('#s17-sample').addEventListener('click',()=>{
    const T=+$('#s17-T').value;const probs=softmax(logits, T);const idx=randChoice(probs);
    // highlight chosen row
    const rows = $$('.pill', host);
    if (rows[idx]) { rows.forEach((r,i)=>{ if(i===idx){ r.style.background='var(--accent2)'; r.style.color='#fff'; setTimeout(()=>{ r.style.background='#fff'; r.style.color='var(--ink)'; }, 800); }}); }
  });
  render();
})();
