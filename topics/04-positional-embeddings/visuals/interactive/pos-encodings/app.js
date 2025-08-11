(function(){
  const dimEl = document.getElementById('dim');
  const lenEl = document.getElementById('len');
  const baseEl = document.getElementById('base');
  const renderBtn = document.getElementById('render');
  const resetBtn = document.getElementById('reset');
  const sinus = document.getElementById('sinus');
  const rope = document.getElementById('rope');
  const sctx = sinus.getContext('2d');
  const rctx = rope.getContext('2d');

  function sinusoidalEmbedding(pos, dim, base){
    const e = new Float32Array(dim);
    for(let i=0;i<dim;i+=2){
      const denom = Math.pow(base, i/dim);
      const angle = pos / denom;
      e[i] = Math.sin(angle);
      if(i+1<dim) e[i+1] = Math.cos(angle);
    }
    return e;
  }

  function renderSinus(dim, n, base){
    const W = sinus.width, H = sinus.height; sctx.clearRect(0,0,W,H);
    const cw = (W-80)/dim, ch = (H-40)/n;
    for(let p=0;p<n;p++){
      const e = sinusoidalEmbedding(p, dim, base);
      for(let d=0; d<dim; d++){
        const v = (e[d]+1)/2; // map [-1,1] to [0,1]
        const hue=(1-v)*220; const light=30+v*40; sctx.fillStyle=`hsl(${hue},80%,${light}%)`;
        sctx.fillRect(60 + d*cw, 20 + p*ch, Math.max(1,cw-1), Math.max(1,ch-1));
      }
      if(p % Math.ceil(n/8) === 0){ sctx.fillStyle='#9bb0c3'; sctx.font='12px system-ui'; sctx.textAlign='right'; sctx.fillText(String(p), 52, 20 + p*ch + ch*0.7); }
    }
    sctx.fillStyle='#9bb0c3'; sctx.font='12px system-ui';
    for(let d=0; d<dim; d+=Math.ceil(dim/8)){
      sctx.save(); sctx.translate(60 + d*cw + cw/2, 10); sctx.rotate(-Math.PI/4); sctx.textAlign='center'; sctx.fillText('d'+d, 0, 0); sctx.restore();
    }
  }

  function ropeAngles(dim, base){
    const angles = new Float32Array(dim/2);
    for(let i=0;i<dim;i+=2){ angles[i/2] = 1/Math.pow(base, i/dim); }
    return angles;
  }

  function rotate2D(x0, x1, theta){
    const c=Math.cos(theta), s=Math.sin(theta); return [x0*c - x1*s, x0*s + x1*c];
  }

  function renderRoPE(dim, n, base){
    const W = rope.width, H = rope.height; rctx.clearRect(0,0,W,H);
    // show first 4 planes, and show BOTH components per plane so rotation is visible
    const planes = Math.min(4, dim/2);
    const rows = planes * 2; // a,b per plane
    const cw = (W-80)/n, ch = (H-40)/rows;
    const thetaScale = ropeAngles(dim, base);

    // start vector with 1 in each first component of every plane, 0 in second
    const x0 = new Float32Array(dim); for(let i=0;i<dim;i+=2) x0[i] = 1;

    for(let p=0; p<n; p++){
      const x = x0.slice();
      for(let i=0;i<dim;i+=2){
        const [a,b] = rotate2D(x[i], x[i+1]||0, p * thetaScale[i/2]);
        x[i]=a; if(i+1<dim) x[i+1]=b;
      }
      for(let k=0;k<planes;k++){
        const a=x[2*k], b=x[2*k+1]||0;
        const va=(a+1)/2, vb=(b+1)/2; // map [-1,1] to [0,1]
        const hueA=(1-va)*220, lightA=30+va*40; rctx.fillStyle=`hsl(${hueA},80%,${lightA}%)`;
        rctx.fillRect(60 + p*cw, 20 + (2*k)*ch, Math.max(1,cw-1), Math.max(1,ch-1));
        const hueB=(1-vb)*220, lightB=30+vb*40; rctx.fillStyle=`hsl(${hueB},80%,${lightB}%)`;
        rctx.fillRect(60 + p*cw, 20 + (2*k+1)*ch, Math.max(1,cw-1), Math.max(1,ch-1));
      }
    }
    // labels
    rctx.fillStyle='#9bb0c3'; rctx.font='12px system-ui';
    for(let p=0;p<n;p+=Math.ceil(n/8)){
      rctx.save(); rctx.translate(60 + p*cw + cw/2, 10); rctx.rotate(-Math.PI/4); rctx.textAlign='center'; rctx.fillText('p'+p, 0, 0); rctx.restore();
    }
    for(let k=0;k<planes;k++){
      rctx.textAlign='right'; rctx.fillText('plane '+k+'-x', 54, 20 + (2*k)*ch + ch*0.7);
      rctx.textAlign='right'; rctx.fillText('plane '+k+'-y', 54, 20 + (2*k+1)*ch + ch*0.7);
    }
  }

  function renderAll(){
    const d=parseInt(dimEl.value,10), n=parseInt(lenEl.value,10), base=parseFloat(baseEl.value);
    renderSinus(d, n, base); renderRoPE(d, Math.min(n, 128), base);
  }
  function reset(){ dimEl.value=64; lenEl.value=64; baseEl.value=10000; renderAll(); }

  renderBtn.addEventListener('click', renderAll);
  resetBtn.addEventListener('click', reset);
  renderAll();
})();
