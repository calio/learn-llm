(function(){
  const dimEl = document.getElementById('dim');
  const headsEl = document.getElementById('heads');
  const causalEl = document.getElementById('causal');
  const viewEl = document.getElementById('view');
  const randomizeBtn = document.getElementById('randomize');
  const resetBtn = document.getElementById('reset');
  const tokensEl = document.getElementById('tokens');
  const sourceEl = document.getElementById('source');
  const canvas = document.getElementById('heatmap');
  const ctx = canvas.getContext('2d');
  const tooltip = document.getElementById('tooltip');

  function tokenize(text){
    return text.trim().split(/\s+/).filter(Boolean).slice(0, 48);
  }

  function randn(){
    // Box-Muller
    let u=0, v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random();
    return Math.sqrt(-2.0*Math.log(u))*Math.cos(2.0*Math.PI*v);
  }

  function randnMatrix(rows, cols){
    const m = new Array(rows);
    for(let r=0;r<rows;r++){ const row=new Float32Array(cols); for(let c=0;c<cols;c++) row[c]=randn()*0.02; m[r]=row; }
    return m;
  }

  function matmul(a, b){
    const r=a.length, k=a[0].length, c=b[0].length;
    const out=new Array(r); for(let i=0;i<r;i++){ const row=new Float32Array(c); for(let j=0;j<c;j++){ let s=0; for(let t=0;t<k;t++) s+=a[i][t]*b[t][j]; row[j]=s; } out[i]=row; }
    return out;
  }

  function softmaxRow(row){
    let m=-Infinity; for(const v of row){ if(v>m) m=v; }
    const ex=new Float32Array(row.length); let s=0; for(let i=0;i<row.length;i++){ const e=Math.exp(row[i]-m); ex[i]=e; s+=e; }
    for(let i=0;i<row.length;i++) ex[i]/=s; return ex;
  }

  function normalizeRow(row){
    let s=0; for(const v of row){ s+=v*v; } s=Math.sqrt(Math.max(1e-9,s)); const out=new Float32Array(row.length); for(let i=0;i<row.length;i++) out[i]=row[i]/s; return out;
  }

  function embedTokens(toks, dim){
    // simple hashed embeddings
    const out=new Array(toks.length);
    for(let i=0;i<toks.length;i++){
      const v=new Float32Array(dim);
      let h=0; for(let c=0;c<toks[i].length;c++){ h=(h*131 + toks[i].charCodeAt(c))>>>0; }
      // fill pseudo-random
      let seed=h; for(let d=0;d<dim;d++){ seed=(1664525*seed+1013904223)>>>0; v[d]=((seed%1000)/1000-0.5)*0.2; }
      out[i]=v;
    }
    return out;
  }

  const state={
    dModel: parseInt(dimEl.value,10),
    heads: parseInt(headsEl.value,10),
    wq:null,wk:null,wv:null,
  };

  function initWeights(){
    const d=state.dModel; state.wq=randnMatrix(d,d); state.wk=randnMatrix(d,d); state.wv=randnMatrix(d,d);
    // populate view select
    while(viewEl.firstChild) viewEl.removeChild(viewEl.firstChild);
    const avgOpt=document.createElement('option'); avgOpt.value='avg'; avgOpt.textContent='Average (all heads)'; viewEl.appendChild(avgOpt);
    for(let h=0; h<state.heads; h++){ const o=document.createElement('option'); o.value=String(h); o.textContent=`Head ${h+1}`; viewEl.appendChild(o); }
  }

  function project(x, w){
    // x: [T, D], w: [D,D] -> [T,D]
    const T=x.length; const D=x[0].length; const out=new Array(T);
    for(let i=0;i<T;i++){
      const row=new Float32Array(D);
      for(let j=0;j<D;j++){
        let s=0; for(let k=0;k<D;k++) s+=x[i][k]*w[k][j];
        row[j]=s;
      }
      out[i]=row;
    }
    return out;
  }

  function splitHeads(x, heads){
    // x: [T, D] -> [heads, T, Hd]
    const T=x.length; const D=x[0].length; const Hd=D/heads; const out=new Array(heads);
    for(let h=0;h<heads;h++){
      const m=new Array(T);
      for(let t=0;t<T;t++){
        const row=new Float32Array(Hd);
        for(let j=0;j<Hd;j++) row[j]=x[t][h*Hd+j];
        m[t]=row;
      }
      out[h]=m;
    }
    return out;
  }

  function attn(q, k, causal){
    const Tq=q.length, Tk=k.length, D=q[0].length; const scale=1/Math.sqrt(D);
    const weights=new Array(Tq);
    for(let i=0;i<Tq;i++){
      const row=new Float32Array(Tk);
      for(let j=0;j<Tk;j++){
        let s=0; for(let d=0;d<D;d++) s+=q[i][d]*k[j][d];
        if(causal && j>i) s=-1e9; // simple causal mask for self-attn case
        row[j]=s*scale;
      }
      weights[i]=softmaxRow(row);
    }
    return weights; // [Tq, Tk]
  }

  function renderHeatmap(A, qToks, kToks){
    const W=canvas.width, H=canvas.height; ctx.clearRect(0,0,W,H);
    const Tq=A.length, Tk=A[0].length;
    const cellW=(W-100)/Tk, cellH=(H-100)/Tq; // leave margins for labels

    // draw cells
    for(let i=0;i<Tq;i++){
      for(let j=0;j<Tk;j++){
        const v=A[i][j];
        const hue=(1-v)*220; const light=30+v*40; ctx.fillStyle=`hsl(${hue},80%,${light}%)`;
        ctx.fillRect(80+j*cellW,20+i*cellH, Math.max(1,cellW-2), Math.max(1,cellH-2));
      }
    }

    // labels
    ctx.fillStyle='#9bb0c3'; ctx.font='12px system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
    for(let j=0;j<Tk;j++){
      const text=kToks[j]||''; ctx.save(); ctx.translate(80+j*cellW+cellW/2, 8); ctx.rotate(-Math.PI/4); ctx.textAlign='center'; ctx.fillText(text,0,0); ctx.restore();
    }
    for(let i=0;i<Tq;i++){
      const text=qToks[i]||''; ctx.textAlign='right'; ctx.fillText(text, 74, 20+i*cellH+cellH*0.7);
    }

    // mouse tooltip
    canvas.onmousemove = (e)=>{
      const rect=canvas.getBoundingClientRect(); const x=e.clientX-rect.left; const y=e.clientY-rect.top;
      const j=Math.floor((x-80)/cellW); const i=Math.floor((y-20)/cellH);
      if(i>=0&&i<Tq&&j>=0&&j<Tk){
        tooltip.style.display='block'; tooltip.style.left=(e.pageX+10)+'px'; tooltip.style.top=(e.pageY+10)+'px';
        tooltip.textContent = `${qToks[i]} → ${kToks[j]} : ${A[i][j].toFixed(3)}`;
      } else { tooltip.style.display='none'; }
    };
    canvas.onmouseleave = ()=> tooltip.style.display='none';
  }

  function compute(){
    state.dModel=parseInt(dimEl.value,10); state.heads=parseInt(headsEl.value,10);
    const toks=tokenize(tokensEl.value); const src=tokenize(sourceEl.value);
    const selfMode = src.length===0;

    // embeddings
    const x=embedTokens(toks, state.dModel); // [Tq, D]
    const y=selfMode? x : embedTokens(src, state.dModel); // [Tk, D]

    // projections
    const q=project(x, state.wq); const k=project(y, state.wk); const v=project(y, state.wv);

    // split heads
    const qh=splitHeads(q, state.heads); const kh=splitHeads(k, state.heads); const vh=splitHeads(v, state.heads);

    // per-head attention weights
    const weights=[]; for(let h=0; h<state.heads; h++){ weights[h]=attn(qh[h].map(normalizeRow), kh[h].map(normalizeRow), causalEl.checked && selfMode); }

    // view selection
    const mode=viewEl.value;
    const qToks=toks, kToks=selfMode ? toks : src;
    if(mode==='avg'){
      // average over heads
      const Tq=weights[0].length, Tk=weights[0][0].length; const A=new Array(Tq); for(let i=0;i<Tq;i++){ const row=new Float32Array(Tk); for(let h=0;h<weights.length;h++){ for(let j=0;j<Tk;j++) row[j]+=weights[h][i][j]; } for(let j=0;j<Tk;j++) row[j]/=weights.length; A[i]=row; }
      renderHeatmap(A,qToks,kToks);
    } else {
      const h=parseInt(mode,10); renderHeatmap(weights[h], qToks, kToks);
    }
  }

  function randomize(){ initWeights(); compute(); }
  function reset(){ dimEl.value=64; headsEl.value=4; causalEl.checked=true; tokensEl.value='the quick brown fox jumps over the lazy dog'; sourceEl.value='le renard brun rapide saute par-dessus le chien paresseux'; randomize(); }

  [dimEl,headsEl,causalEl,viewEl,tokensEl,sourceEl].forEach(el=> el.addEventListener('input', compute));
  randomizeBtn.addEventListener('click', randomize);
  resetBtn.addEventListener('click', reset);

  initWeights(); compute();
})();
