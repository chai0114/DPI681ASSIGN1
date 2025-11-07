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

/* === Scene 3–4: n-gram (10-sentence corpus + calculation table) === */
(function(){
  const sketch = $('#s3-sketch');
  const tableHost = $('#s3-table');
  const corpusHost = $('#s3-corpus');

  // 10-sentence toy corpus (lowercase)
  const corpusSentences = [
    'i like pizza',
    'i like pasta',
    'you like pizza',
    'i like tea very much',
    'you love pasta',
    'they like tea',
    'we like pizza and pasta',
    'i love pizza very much',
    'you like pasta very much',
    'we love tea and pizza'
  ];
  const corpus = corpusSentences.map(s => s.trim().split(/\s+/));
  const vocab = Array.from(new Set(corpus.flat()));

  // Show corpus for transparency
  function renderCorpus(){
    const det = document.createElement('details');
    det.innerHTML = '<summary>Corpus (10 sentences)</summary>';
    const ol = document.createElement('ol');
    corpusSentences.forEach(s => { const li = document.createElement('li'); li.textContent = s; ol.appendChild(li); });
    det.appendChild(ol);
    corpusHost.innerHTML = '';
    corpusHost.appendChild(det);
  }

  // Bars (absolute probabilities by default)
  function drawBars(items){
    drawBarRow(sketch, items, { normalize:false });
  }

  // Nice status
  function status(msg, kind='ok'){
    showStatus(sketch, msg, kind);
  }

  // Build calculation table (raw counts + smoothed prob)
  function buildTable(rawCounts, rawTotal, contextStr, n, matches, smoothingOn){
    tableHost.innerHTML = '';

    const head = document.createElement('div');
    head.className = 'status-box status-ok';
    const unseenCount = Math.max(0, vocab.length - Object.keys(rawCounts).length);
    head.textContent =
      `n=${n} • context="${contextStr}" • matches=${rawTotal}` +
      (smoothingOn ? ` • Laplace(+1): |V|=${vocab.length}, unseen=${unseenCount}` : '');
    tableHost.appendChild(head);

    const explain = document.createElement('p');
    explain.className = 'small';
    explain.innerHTML = smoothingOn
      ? `P(next|ctx) = (count(ctx→w) + 1) / (Σ counts + |V|). Unseen words also get +1; their combined mass is <span class="mono">${unseenCount}/(${rawTotal}+${vocab.length})</span>.`
      : `P(next|ctx) = count(ctx→w) / Σ counts.`;
    tableHost.appendChild(explain);

    const table = document.createElement('table');
    table.style.width='100%'; table.style.borderCollapse='collapse';
    const thead=document.createElement('thead');
    const trh=document.createElement('tr');
    ['Next word','Raw count','Probability'].forEach(h=>{
      const th=document.createElement('th');
      th.textContent=h; th.style.textAlign='left';
      th.style.borderBottom='2px solid var(--ink)'; th.style.padding='.25rem';
      trh.appendChild(th);
    });
    thead.appendChild(trh); table.appendChild(thead);

    const denom = smoothingOn ? (rawTotal + vocab.length) : rawTotal;
    const tbody=document.createElement('tbody');
    Object.entries(rawCounts).sort((a,b)=>b[1]-a[1]).forEach(([w,c])=>{
      const tr=document.createElement('tr');
      const tdw=document.createElement('td'); tdw.textContent=w; tdw.style.padding='.25rem';
      const tdc=document.createElement('td'); tdc.textContent=c; tdc.style.padding='.25rem';
      const prob = smoothingOn ? (c + 1) / denom : c / denom;
      const tdp=document.createElement('td'); tdp.textContent = `${prob.toFixed(3)} (${(prob*100).toFixed(1)}%)`; tdp.style.padding='.25rem';
      tr.append(tdw, tdc, tdp); tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    tableHost.appendChild(table);

    const det=document.createElement('details');
    det.innerHTML='<summary>Show matched n-grams</summary>';
    const ul=document.createElement('ul'); ul.style.marginTop='.25rem';
    matches.forEach(m=>{
      const li=document.createElement('li');
      const ctx = m.window.slice(0, n-1).join(' ');
      li.innerHTML = `<span class="mono">[${ctx}] → ${m.next}</span>`;
      ul.appendChild(li);
    });
    det.appendChild(ul);
    tableHost.appendChild(det);
  }

  function computeNGrams(){
    const n = +$('#s3-n').value;
    const contextStr = $('#s3-context').value.trim().toLowerCase();
    const smoothing = $('#s3-smooth').checked;
    const context = contextStr ? contextStr.split(/\s+/) : [];
    const contextLen = n - 1;

    // validations
    if (contextLen <= 0) { status(`Error: n must be 2 or greater.`, 'err'); tableHost.innerHTML=''; return; }
    if (context.length !== contextLen) { status(`Error: Context must be ${contextLen} word(s) for n=${n}.`, 'err'); tableHost.innerHTML=''; return; }

    // raw counting (no smoothing here)
    const rawCounts = {};
    const matches = [];
    let rawTotal = 0;

    corpus.forEach(tokens => {
      for (let i = 0; i <= tokens.length - n; i++) {
        const window = tokens.slice(i, i + n);
        const winCtx = window.slice(0, contextLen);
        if (winCtx.every((w,j)=>w === context[j])) {
          const nextWord = window[contextLen];
          rawCounts[nextWord] = (rawCounts[nextWord] || 0) + 1;
          rawTotal++;
          matches.push({window, next: nextWord});
        }
      }
    });

    if (rawTotal === 0) {
      status(`No counts found for context "${contextStr}" (Data sparsity). ${smoothing ? 'Smoothing will still assign probability mass to unseen words (uniform +1).' : 'Try smaller n or a different context.'}`, 'ok');
      tableHost.innerHTML = '';
      return;
    }

    // probabilities
    const denom = smoothing ? (rawTotal + vocab.length) : rawTotal;
    const observedItems = Object.entries(rawCounts)
      .map(([w,c]) => ({ word: w, p: (smoothing ? (c+1) : c) / denom }))
      .sort((a,b)=>b.p - a.p);

    // add aggregated "unseen" bar if smoothing
    if (smoothing) {
      const unseenCount = Math.max(0, vocab.length - Object.keys(rawCounts).length);
      if (unseenCount > 0) {
        const unseenMass = unseenCount / denom;
        observedItems.push({ word: '(unseen words)', p: unseenMass });
      }
    }

    drawBars(observedItems);
    buildTable(rawCounts, rawTotal, contextStr, n, matches, smoothing);
  }

  // events
  $('#s3-calc').addEventListener('click', computeNGrams);
  $('#s3-n').addEventListener('input', () => { $('#s3-nv').textContent = $('#s3-n').value; computeNGrams(); });

  // init
  renderCorpus();
  computeNGrams();
})();


/* === Scene 5–6: CBOW embeddings (short sentences, average vector) === */
(function(){
  const map = $('#s5-map'); const ctx = map.getContext('2d');
  const barsHost = $('#s5-bars'); const tableHost = $('#s5-table');

  // Small 2D embedding lexicon tuned so the examples behave intuitively.
  // Units are arbitrary in [0,1], but cosine similarity will drive predictions.
  const vec = {
    // sonnet example
    "shall":[0.50,0.50],"i":[0.50,0.50],"compare":[0.60,0.55],"thee":[0.56,0.60],"to":[0.50,0.50],"a":[0.50,0.50],
    "summer's":[0.84,0.84],
    "day":[0.82,0.79],"night":[0.25,0.80],"river":[0.15,0.30],"story":[0.40,0.25],"temperate":[0.78,0.76],

    // directions example
    "i":[0.50,0.50],"should":[0.52,0.48],"turn":[0.72,0.35],"right":[0.88,0.30],
    "at":[0.50,0.50],"the":[0.50,0.50],"light":[0.86,0.32],
    "right?":[0.88,0.30],"now":[0.60,0.45],"left":[0.15,0.30],"today":[0.55,0.55],"slowly":[0.58,0.40],

    // punctuation fallback
    ",":[0.50,0.50], "?":[0.50,0.50]
  };
  const STOP = new Set(["the","a","an","to","of","and","i","at","shall"]);

  const EXAMPLES = {
    sonnet: {
      context: ["shall","i","compare","thee","to","a","summer's"],
      candidates: ["day","night","river","story","temperate"],
      label: 'Shall I compare thee to a summer’s ___'
    },
    directions: {
      context: ["i","should","turn","right","at","the","light"],
      candidates: ["right?","now","left","today","slowly"],
      label: 'I should turn right at the light ___'
    }
  };

  // ---------- math helpers ----------
  function dot(a,b){ return a[0]*b[0] + a[1]*b[1]; }
  function norm(a){ return Math.hypot(a[0], a[1]); }
  function cosine(a,b){
    const na = norm(a), nb = norm(b);
    if(na===0 || nb===0) return 0;
    return dot(a,b)/(na*nb);
  }
  function mean(vectors){
    if(vectors.length===0) return [0,0];
    const s = vectors.reduce((acc,v)=>[acc[0]+v[0],acc[1]+v[1]],[0,0]);
    return [ s[0]/vectors.length, s[1]/vectors.length ];
  }
  function softmax(xs, T=0.5){ // slight sharpening so top candidate is clear
    const scaled = xs.map(x => x / T);
    const m = Math.max(...scaled);
    const ex = scaled.map(x => Math.exp(x - m));
    const s = ex.reduce((a,b)=>a+b,0);
    return ex.map(x => x/s);
  }

  // ---------- state ----------
  let exampleKey = $('#s5-example') ? $('#s5-example').value : 'sonnet';
  let included = new Set(EXAMPLES[exampleKey].context); // click toggles

  // ---------- drawing ----------
  const PAD = 36; const W = map.width; const H = map.height;
  function toXY([x,y]){ return [PAD + x*(W-2*PAD), H - PAD - y*(H-2*PAD)]; }

  function drawScatter(ctxWords, candidates, mu, topWord){
    ctx.clearRect(0,0,W,H);
    // grid
    ctx.strokeStyle='#ddd'; ctx.lineWidth=1;
    for(let i=0;i<5;i++){
      const x=PAD+i*(W-2*PAD)/4; ctx.beginPath(); ctx.moveTo(x,PAD); ctx.lineTo(x,H-PAD); ctx.stroke();
      const y=PAD+i*(H-2*PAD)/4; ctx.beginPath(); ctx.moveTo(PAD,y); ctx.lineTo(W-PAD,y); ctx.stroke();
    }
    ctx.strokeStyle='var(--ink)'; ctx.lineWidth=2; ctx.strokeRect(PAD,PAD,W-2*PAD,H-2*PAD);

    // context words
    ctxWords.forEach(w=>{
      const v = vec[w]; if(!v) return;
      const [x,y]=toXY(v);
      ctx.fillStyle = included.has(w) ? '#2a7bd2' : '#999';
      ctx.beginPath(); ctx.arc(x,y,6,0,Math.PI*2); ctx.fill();
      ctx.fillStyle='#000'; ctx.font='bold 13px ui-sans-serif';
      ctx.fillText(w, x+8, y-8);
    });

    // candidates (gray)
    candidates.forEach(w=>{
      const v = vec[w]; if(!v) return;
      const [x,y]=toXY(v);
      ctx.strokeStyle='#666'; ctx.lineWidth=2;
      ctx.beginPath(); ctx.arc(x,y,7,0,Math.PI*2); ctx.stroke();
      ctx.fillStyle='#000'; ctx.font='bold 13px ui-sans-serif';
      ctx.fillText(w, x+8, y-8);
    });

    // mean μ (orange star)
    const [mx,my]=toXY(mu);
    ctx.fillStyle='#f08b2f';
    ctx.beginPath();
    for(let i=0;i<5;i++){
      const ang = -Math.PI/2 + i*2*Math.PI/5;
      const r = (i%2===0 ? 10 : 5);
      const x = mx + r*Math.cos(ang), y = my + r*Math.sin(ang);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.closePath(); ctx.fill();
    ctx.fillStyle='#000'; ctx.font='bold 12px ui-sans-serif'; ctx.fillText('μ', mx+10, my-8);

    // ring top prediction
    if(topWord && vec[topWord]){
      const [tx,ty]=toXY(vec[topWord]);
      ctx.strokeStyle='#f08b2f'; ctx.lineWidth=3;
      ctx.beginPath(); ctx.arc(tx,ty,12,0,Math.PI*2); ctx.stroke();
    }
  }

  function drawBars(items){
    drawBarRow(barsHost, items, { normalize:false });
  }

  function renderTable(ctxWords, mu, scores){
    const parts = [];
    parts.push(`Context (included): ${ctxWords.filter(w=>included.has(w)).join(' ') || '(none)'}`);
    parts.push(`μ = [${mu[0].toFixed(2)}, ${mu[1].toFixed(2)}]`);
    parts.push(`Cosines: ${scores.map(s=>`${s.word}:${s.cos.toFixed(2)}`).join('  |  ')}`);
    tableHost.textContent = parts.join('  •  ');
  }

  // ---------- interaction ----------
  function currentExample(){ return EXAMPLES[exampleKey]; }
  function activeContextWords(){
    const ignoreStops = $('#s5-ignore-stops')?.checked;
    const base = currentExample().context.slice();
    return base.filter(w => included.has(w) && !(ignoreStops && STOP.has(w)));
  }

  function compute(){
    const ex = currentExample();
    const ctxWords = activeContextWords();
    const ctxVecs = ctxWords.map(w => vec[w] || [0.5,0.5]);
    const mu = mean(ctxVecs);

    const cosines = ex.candidates.map(w => ({
      word: w,
      cos: cosine(mu, vec[w] || [0.5,0.5])
    }));
    const probs = softmax(cosines.map(c=>c.cos), 0.5);
    const items = cosines
      .map((c,i)=>({ word:c.word, p:probs[i], cos:c.cos }))
      .sort((a,b)=>b.p-a.p);

    drawScatter(ex.context, ex.candidates, mu, items[0]?.word);
    drawBars(items);
    renderTable(ctxWords, mu, items);
  }

  function resetExample(){
    included = new Set(currentExample().context);
    compute();
  }

  // clicks on points to toggle inclusion
  map.addEventListener('click', ev=>{
    const r = map.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    // hit test only for context words (not candidates)
    const ex = currentExample();
    for(const w of ex.context){
      const v = vec[w]; if(!v) continue;
      const [px,py] = toXY(v);
      if(Math.hypot(px - x, py - y) < 10){
        if(included.has(w)) included.delete(w); else included.add(w);
        compute(); return;
      }
    }
  });

  // wire UI
  $('#s5-example').addEventListener('change', (e)=>{ exampleKey = e.target.value; resetExample(); });
  $('#s5-ignore-stops').addEventListener('change', compute);
  $('#s5-predict').addEventListener('click', compute);
  $('#s5-reset-emb').addEventListener('click', resetExample);

  // initial
  resetExample();
})();


/* === Scene 7: 2×2 network diagram (inputs→hidden→outputs) === */
(function(){
  const canvas = $('#s7-net');
  const ctx = canvas.getContext('2d');
  const barsHost = $('#s7-bars');

  const labels = ['Vertical','Horizontal','Diagonal','None'];

  // Pretrained tiny model (same values as 이전 예제)
  const model = {
    // Input[4] -> Hidden[3]
    W_ih: [
      [ 2.0, -1.0,  1.0], // x1 (top-left)
      [-1.0,  2.0, -1.0], // x2 (top-right)
      [ 2.0, -1.0, -1.0], // x3 (bottom-left)
      [-1.0,  2.0,  1.0]  // x4 (bottom-right)
    ],
    b_h: [0.2, 0.2, 0.1],
    // Hidden[3] -> Output[4]
    W_ho: [
      [ 3.0, -1.0, -1.0, -1.0], // h1 → (V,H,D,None)
      [-1.0,  3.0, -1.0, -1.0], // h2 →
      [-1.0, -1.0,  3.0, -1.0]  // h3 →
    ],
    b_o: [0,0,0,0]
  };

  // Input presets
  const PRESETS = {
    Vertical:   [1,0,1,0],
    Horizontal: [1,1,0,0],
    Diagonal:   [1,0,0,1],
    None:       [0,0,0,0],
    All:        [1,1,1,1]
  };

  // Math helpers
  const relu = x => Math.max(0, x);
  function softmax(arr, T=1){
    const scaled = arr.map(v => v / T);
    const m = Math.max(...scaled);
    const ex = scaled.map(v => Math.exp(v - m));
    const s = ex.reduce((a,b)=>a+b,0);
    return ex.map(v => v/s);
  }

  // Layout
  const W = canvas.width, H = canvas.height;
  const Lx = 90, Cx = 280, Rx = 470; // x-positions: inputs, hidden, outputs
  const Iy = [90, 160, 230, 300];    // inputs: 2×2 squares (TL,TR,BL,BR) projected vertically
  const Hy = [110, 200, 290];        // hidden nodes
  const Oy = [90, 160, 230, 300];    // outputs

  // Node drawing
  function drawInput(i, on){
    // Map i: 0 TL, 1 TR, 2 BL, 3 BR to positions (two rows)
    const grid = [
      [Lx, Iy[0]], [Lx, Iy[1]], [Lx, Iy[2]], [Lx, Iy[3]]
    ];
    const [x,y] = grid[i];
    ctx.fillStyle = on ? 'var(--accent)' : '#fff';
    ctx.strokeStyle = 'var(--ink)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.rect(x-14, y-14, 28, 28);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#000'; ctx.font = 'bold 12px ui-sans-serif';
    ctx.fillText(`x${i+1}`, x-22, y-20);
  }
  function drawCircle(x,y,r,fill,stroke,label){
    ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); ctx.stroke();
    if(label){ ctx.fillStyle='#000'; ctx.font='bold 12px ui-sans-serif'; ctx.fillText(label, x-r-20, y-r-10); }
  }

  function linkColor(w, active, showNeg){
    if(!active) return 'rgba(120,120,120,0.25)';
    if(w === 0) return 'rgba(120,120,120,0.6)';
    const s = Math.min(1, Math.abs(w)/3.0); // scale
    if(w > 0) return `rgba(42,123,210,${0.25 + 0.75*s})`; // blue(+)
    if(showNeg) return `rgba(195,51,51,${0.25 + 0.75*s})`; // red(−)
    return 'rgba(120,120,120,0.35)';
  }

  function drawLink(x0,y0,x1,y1,w,active,showNeg){
    ctx.strokeStyle = linkColor(w, active, showNeg);
    ctx.lineWidth = 1 + Math.min(5, Math.abs(w));
    ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke();
  }

  function predict(x){
    // hidden pre-acts
    const z = [0,0,0];
    for(let h=0; h<3; h++){
      z[h] = model.b_h[h];
      for(let i=0; i<4; i++) z[h] += x[i]*model.W_ih[i][h];
    }
    const h = z.map(relu);
    // outputs
    const y = [0,0,0,0];
    for(let o=0; o<4; o++){
      y[o] = model.b_o[o];
      for(let k=0; k<3; k++) y[o] += h[k]*model.W_ho[k][o];
    }
    const p = softmax(y);
    return { z, h, y, p };
  }

  function drawDiagram(x, showW, showNeg){
    ctx.clearRect(0,0,W,H);

    // 1) edges: input→hidden
    for(let i=0;i<4;i++){
      const x0 = Lx, y0 = Iy[i];
      for(let h=0;h<3;h++){
        const x1 = Cx, y1 = Hy[h];
        const w = model.W_ih[i][h];
        drawLink(x0,y0,x1,y1, w, x[i]===1, showNeg);
      }
    }
    // 2) edges: hidden→output (always draw; active scaled by hidden>0)
    const tmp = predict(x);
    for(let h=0;h<3;h++){
      const x0 = Cx, y0 = Hy[h];
      for(let o=0;o<4;o++){
        const x1 = Rx, y1 = Oy[o];
        const w = model.W_ho[h][o];
        const active = tmp.h[h] > 0;
        drawLink(x0,y0,x1,y1, w, active, showNeg);
      }
    }

    // 3) nodes
    // inputs
    for(let i=0;i<4;i++) drawInput(i, x[i]===1);

    // hidden (show z & h)
    for(let h=0;h<3;h++){
      drawCircle(Cx, Hy[h], 14, '#fff', 'var(--ink)', `h${h+1}`);
      // annotations
      ctx.fillStyle='#000'; ctx.font='11px ui-sans-serif';
      const { z, h:act } = tmp;
      ctx.fillText(`z=${z[h].toFixed(1)}`, Cx+18, Hy[h]-2);
      ctx.fillText(`ReLU=${act[h].toFixed(1)}`, Cx+18, Hy[h]+12);
    }

    // outputs (show logits & class label)
    for(let o=0;o<4;o++){
      drawCircle(Rx, Oy[o], 14, '#fff', 'var(--ink)', labels[o][0]); // first letter tag
      ctx.fillStyle='#000'; ctx.font='11px ui-sans-serif';
      ctx.fillText(labels[o], Rx+18, Oy[o]-2);
      ctx.fillText(`logit=${tmp.y[o].toFixed(1)}`, Rx+18, Oy[o]+12);
    }

    // 4) weight labels (optional)
    if(showW){
      ctx.fillStyle='#333'; ctx.font='11px ui-sans-serif';
      ctx.fillText('W_ih (4×3) + b_h', Cx-40, 40);
      ctx.fillText('W_ho (3×4) + b_o', Rx-40, 40);
    }

    // 5) bars for probabilities
    const items = labels.map((w,i)=>({word:w, p: tmp.p[i]})).sort((a,b)=>b.p-a.p);
    drawBarRow(barsHost, items, { normalize:false });
  }

  function setPattern(name){
    const x = PRESETS[name];
    const showW = $('#s7-show-w').checked;
    const showNeg = $('#s7-show-neg').checked;
    drawDiagram(x, showW, showNeg);
  }

  // Wire up
  ['Vertical','Horizontal','Diagonal','None','All'].forEach(k=>{
    $('#s7-'+k.toLowerCase()).addEventListener('click',()=>setPattern(k));
  });
  $('#s7-show-w').addEventListener('change', ()=>setPattern(currentPattern));
  $('#s7-show-neg').addEventListener('change', ()=>setPattern(currentPattern));

  let currentPattern = 'Vertical';
  function setPattern(name){ currentPattern = name; const x = PRESETS[name];
    const showW = $('#s7-show-w').checked;
    const showNeg = $('#s7-show-neg').checked;
    drawDiagram(x, showW, showNeg);
  }

  // initial
  setPattern('Vertical');
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
