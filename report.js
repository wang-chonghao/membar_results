/* Offline report: all experiment data and assets are local. */
(() => {
  'use strict';
  const data = window.MEMBAR_REPORT;
  const view = document.getElementById('view');
  const title = document.getElementById('page-title');
  const nav = document.getElementById('case-nav');
  const number = value => value == null ? '<span class="dash">—</span>' : Number(value).toLocaleString('en-US');
  const percent = value => value == null ? '<span class="dash">—</span>' : `${value.toFixed(2)}%`;
  const escape = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const defaultVariant = item => item.variants.find(v => v.id === item.default);
  let chart;
  let selected;
  let overlay = false;

  Chart.defaults.font.family = 'system-ui, "Microsoft YaHei", sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.color = '#63707d';
  nav.innerHTML = '<a href="#overview" class="nav-link" data-case="overview">全部结果<small>时间、spill 与优化收益</small></a>' + data.cases.map(c => `<a class="nav-link" href="#${c.id}" data-case="${c.id}">${escape(c.title)}<small>${escape(c.subtitle)}</small></a>`).join('');

  function draw(series, bar = false) {
    if (chart) chart.destroy();
    const context = document.getElementById('ipc-chart');
    const sets = series.map(s => ({label:s.label, data:bar ? s.values : s.samples.map((y,x) => ({x,y})),
      borderColor:s.color, backgroundColor:s.color, borderWidth:bar ? 0 : 1.6,
      pointRadius:0, pointHitRadius:5, tension:0, fill:false}));
    chart = new Chart(context, {type:bar ? 'bar' : 'line',
      data:{labels:bar ? data.cases.map(c=>c.title.replace(' 三段循环',' 切分')) : undefined, datasets:sets},
      options:{responsive:true,maintainAspectRatio:false,animation:false,normalized:!bar,
        interaction:{mode:bar ? 'index' : 'nearest',intersect:false,axis:'x'},
        plugins:{legend:{position:'top',align:'start',labels:{boxWidth:12,boxHeight:9,padding:17}},
          tooltip:{callbacks:{title:items=>bar ? items[0].label : `相对 cycle ${Math.round(items[0].parsed.x)}`}}},
        scales:{x:{type:bar ? 'category' : 'linear',grid:{display:false},title:{display:!bar,text:'相对首条计算完成的周期'},ticks:{maxTicksLimit:8}},
          y:{beginAtZero:true,grid:{color:'#e8edf0'},suggestedMax:bar ? undefined : 2,title:{display:true,text:bar ? 'VF 周期' : '计算完成 IPC'}}}}});
  }

  function overview() {
    title.textContent = 'Membar 实验结果';
    const variants = data.cases.flatMap(c=>c.variants);
    const rows = data.cases.map(c=>{
      const v=defaultVariant(c);
      return `<tr><td><a href="#${c.id}">${escape(c.title)}</a><span class="subtle">${escape(v.label)}</span></td><td class="num">${number(v.camodel)}</td><td class="num">${number(v.global_cycles)}</td><td class="num">${number(v.local_cycles)}</td><td class="num">${percent(v.accuracy)}</td><td class="num">${v.speedup ? v.speedup.toFixed(2)+'x' : '—'}</td></tr>`;
    }).join('');
    view.innerHTML = `<p class="lede intro">同步空泡、架构寄存器 spill，以及通过循环切分恢复性能的 A5 实验。</p>
      <p class="shape">CAModel 为实际硬件模型执行结果；局部依赖周期为保留数据约束后的实验预测。</p>
      <div class="stats"><div class="stat"><label>实验 case</label><strong>${data.cases.length}</strong><small>阶段同步 / 向量 spill / 谓词 spill / 循环切分</small></div><div class="stat"><label>有效源码版本</label><strong>${variants.length}</strong><small>均包含 CCE 和两份 CAModel 原始日志</small></div><div class="stat"><label>数值校验</label><strong>全部通过</strong><small>来自原实验的 golden 或独立验证记录</small></div><div class="stat green"><label>Softmax 模型内收益</label><strong>1.22x</strong><small>638 → 524 cycle，同一动态指令流</small></div></div>
      <section class="section"><div class="section-head"><h2>代表版本时间对比</h2><a href="comparison.csv" download>下载全部版本</a></div><div class="table-scroll"><table class="overview-table"><thead><tr><th>Case / 版本</th><th class="num">CAModel</th><th class="num">全局预测</th><th class="num">局部预测</th><th class="num">全局精度</th><th class="num">模型内加速</th></tr></thead><tbody>${rows}</tbody></table></div></section>
      <section class="section"><div class="section-head"><h2>代表版本周期</h2><button class="text-link" id="export-plot">导出图表 PNG</button></div><div class="plot-surface"><div class="plot overview-plot"><canvas id="ipc-chart" aria-label="代表版本周期对比图" role="img"></canvas></div></div></section>
      <div class="note"><p>寄存器溢出的 7 个版本均已提供 <strong>hardware_equivalent.cce</strong>：这是根据实际硬件日志还原的 VF 执行指令流的等效 CCE，已补齐编译器插入的保存、重载和 mem_bar。<a href="hardware_equivalent_index.json">查看等效代码清单</a>。</p><p>GeLU Poly、SwiGLU Grad、AdamApplyOne 的 spill 收益采用编译后 PC 回放进行同流 A/B；GeLU Grad 保留显式 spill 实验源码及按对应日志还原的代码。原始算法源码仍单独保留。</p></div>`;
    draw([{label:'CAModel 实测',color:'#344251',values:data.cases.map(c=>defaultVariant(c).camodel)},
      {label:'VfSim 全局同步',color:'#2376b8',values:data.cases.map(c=>defaultVariant(c).global_cycles)},
      {label:'VfSim 局部依赖',color:'#18846d',values:data.cases.map(c=>defaultVariant(c).local_cycles)}],true);
    bindExport('membar_overview');
  }

  function tableRow(v) {
    return `<tr class="${v.id===selected.id ? 'selected' : ''}" data-variant="${v.id}"><td><button class="table-link" data-select="${v.id}">${escape(v.label)}</button></td><td class="num">${number(v.camodel)}</td><td class="num">${number(v.source_cycles)}</td><td class="num">${number(v.global_cycles)}</td><td class="num">${number(v.local_cycles)}</td><td class="num">${percent(v.accuracy)}</td><td class="num">${v.speedup ? v.speedup.toFixed(2)+'x' : '—'}</td><td class="num">${v.vector_store}/${v.vector_load}</td><td class="num">${v.predicate_store}/${v.predicate_load}</td><td class="num">${v.membars}</td></tr>`;
  }

  function renderCase(item, variant = defaultVariant(item)) {
    selected=variant;
    title.textContent=item.title;
    const v=selected;
    const hardware=v.hardware_equivalent ? `<section class="hardware-code"><h2>实际底层硬件执行指令流 · 等效 CCE</h2><p><a class="hardware-file" href="${v.hardware_equivalent.path}">hardware_equivalent.cce</a><a href="${v.hardware_equivalent.mapping}">PC 与指令数量核对</a></p><p>这份代码对应本版本 CAModel 日志中实际执行的 VF 计算、UB 读写及同步序列，已补入溢出后编译器插入的 VLDS/VSTS、${v.predicate_store ? 'PLDS/PSTS、' : ''}mem_bar，保留指令顺序、寄存器复用和循环次数。</p><p class="muted">标量地址计算效果已折入指针及立即数；原始标量指令仍见日志。表中预测时间沿用原实验的预测输入，新归档的等效代码未重新编译或重新预测。</p></section>` : '';
    view.innerHTML=`<p class="lede">${escape(item.headline)}</p><p class="shape">${escape(item.shape)}</p>
      <div class="stats"><div class="stat"><label>CAModel 实测</label><strong>${number(v.camodel)}</strong><small>${escape(v.label)}</small></div><div class="stat"><label>VfSim 全局同步</label><strong>${number(v.global_cycles)}</strong><small>${v.accuracy==null ? '该源码预测缺少编译器 spill' : '精度 '+percent(v.accuracy)}</small></div><div class="stat green"><label>VfSim 局部依赖</label><strong>${number(v.local_cycles)}</strong><small>${v.speedup ? '模型内加速 '+v.speedup.toFixed(2)+'x' : '此版本没有同流局部 A/B 记录'}</small></div><div class="stat"><label>数值校验</label><strong>通过</strong><small>原实验 golden / 独立核查</small></div></div>
      ${hardware}
      <section class="section"><div class="section-head"><h2>各版本结果</h2><a href="${item.id}/summary.json">结果与来源</a></div><div class="table-scroll"><table class="result-table"><thead><tr><th>版本</th><th class="num">CAModel</th><th class="num">源码预测</th><th class="num">全局预测</th><th class="num">局部预测</th><th class="num">全局精度</th><th class="num">加速</th><th class="num">Vector 写/读</th><th class="num">Predicate 写/读</th><th class="num">Membar</th></tr></thead><tbody>${item.variants.map(tableRow).join('')}</tbody></table></div><p class="plot-caption">spill 与 Membar 数量均为动态次数。全局预测优先采用含 spill 的回放；未 spill 档位使用源码输入。源码预测缺失 spill 的档位不计算全局精度。</p></section>
      <section class="section"><div class="section-head"><h2>计算完成 IPC</h2><div class="toolbar"><label for="variant-select" class="muted">版本</label><select id="variant-select">${item.variants.map(x=>`<option value="${x.id}" ${x.id===v.id ? 'selected' : ''}>${escape(x.label)}</option>`).join('')}</select></div></div>
      <div class="section-head">${item.overlay ? `<div class="mode-picker"><label><input type="radio" name="plot-mode" value="single" ${!overlay?'checked':''}>本版本模式对比</label><label><input type="radio" name="plot-mode" value="overlay" ${overlay?'checked':''}>循环切分实测对比</label></div>` : '<span></span>'}<button class="text-link" id="export-plot">导出图表 PNG</button></div>
      <div class="plot-surface"><div class="plot"><canvas id="ipc-chart" role="img" aria-label="${escape(item.title)} 计算完成 IPC"></canvas></div><p class="plot-caption">${escape(data.ipc_method)} 共用时间标尺保留不同阶段间的空泡。</p></div></section>
      <div class="chips"><span>Vector spill：<strong>${v.vector_store} 写 / ${v.vector_load} 读</strong></span><span>Predicate spill：<strong>${v.predicate_store} 写 / ${v.predicate_load} 读</strong></span><span>动态 Membar：<strong>${v.membars}</strong></span></div>
      <div class="note"><p>${escape(v.note)}</p>${item.notes.map(n=>`<p>${escape(n)}</p>`).join('')}</div>
      <section class="section"><h2>源码与 CAModel 日志</h2><div class="files"><div><h3>CCE 与结果</h3><ul class="file-list">${v.sources.map(s=>`<li><a href="${s.path}">${escape(s.label)} · ${escape(s.path.split('/').pop())}</a></li>`).join('')}<li><a href="${item.id}/${v.id}/figures/ipc.png">本版本 IPC 图 · PNG</a></li>${item.overlay ? '<li><a href="gelu_grad_three_stage/figures/camodel_u1_u4_three_stage.png">U1 / U4 / 三段循环实测叠图 · PNG</a></li>' : ''}</ul><a href="${item.id}/${v.id}/result.json">本版本元数据与 IPC 数据</a></div><div><h3>原始日志</h3><ul class="file-list">${v.logs.map(p=>`<li><a href="${p}">${escape(p.split('/').pop())}</a></li>`).join('')}</ul><details><summary>校验与实验记录</summary><ul class="file-list">${v.evidence.map(p=>`<li><a href="${p}">${escape(p.split('/').pop())}</a></li>`).join('')}</ul></details></div></div></section>`;
    updateCasePlot(item);
    view.querySelectorAll('[data-select]').forEach(b=>b.addEventListener('click',()=>renderCase(item,item.variants.find(x=>x.id===b.dataset.select))));
    document.getElementById('variant-select').addEventListener('change',e=>renderCase(item,item.variants.find(x=>x.id===e.target.value)));
    view.querySelectorAll('[name="plot-mode"]').forEach(r=>r.addEventListener('change',e=>{overlay=e.target.value==='overlay';updateCasePlot(item);}));
    bindExport(`${item.id}_${v.id}`);
  }

  function updateCasePlot(item) {
    let series=selected.series;
    if(item.overlay && overlay) {
      const colors=['#344251','#2376b8','#18846d'];
      series=item.variants.map((v,i)=>({...v.series.find(s=>s.key==='camodel'),label:v.label,color:colors[i]}));
    }
    draw(series);
  }

  function bindExport(name) {
    document.getElementById('export-plot').addEventListener('click',()=>{
      const canvas=document.getElementById('ipc-chart');
      const clean=document.createElement('canvas');clean.width=canvas.width;clean.height=canvas.height;
      const ctx=clean.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,clean.width,clean.height);ctx.drawImage(canvas,0,0);
      const link=document.createElement('a');link.download=name+'.png';link.href=clean.toDataURL('image/png');link.click();
    });
  }

  function route() {
    const id=location.hash.slice(1)||'overview';
    const item=data.cases.find(c=>c.id===id);
    nav.querySelectorAll('a').forEach(a=>{const active=a.dataset.case===(item ? id : 'overview');a.classList.toggle('active',active);active ? a.setAttribute('aria-current','page') : a.removeAttribute('aria-current');});
    overlay=!!item?.overlay;
    item ? renderCase(item) : overview();
  }
  window.addEventListener('hashchange',route);
  route();
  window.MEMBAR_CHART = () => chart;
})();
