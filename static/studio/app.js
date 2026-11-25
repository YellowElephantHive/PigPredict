import {Decoder, deltaE, rgbCss, labToRgb, validateTarget} from './engine.js';

const $ = id => document.getElementById(id);
const english = Object.fromEntries([...document.querySelectorAll('[data-i18n]')].map(el => [el.dataset.i18n, el.innerHTML]));
const chinese = {
  skip:'跳至配色工作台', workspace:'配色工作台', methodNav:'配色背後的技術', eyebrow:'染整製程知識，結合神經網路。',
  title:'從一個目標色，<br>找到染料配方。', intro:'將目標色轉換成染料配方。比較候選組合、微調染料濃度，即時探索模型預測的變化。',
  original:'使用專案原始訓練模型', dyes:'種染料', fabrics:'種布料', layers:'層全連接網路', start:'從目標色開始', targetTitle:'設定配色目標',
  targetColor:'目標色', lightness:'明度', redGreen:'綠色／紅色', blueYellow:'藍色／黃色', substrate:'布料種類', sampleTargets:'試用實測樣本',
  sampleCaption:'取自專案歷史資料', find:'搜尋染料配方', loading:'正在載入原始模型…', targetNote:'選擇布料後開始搜尋。候選染料組合來自歷史配方，再由模型反推濃度。',
  resultEyebrow:'讓色彩成為可操作的配方', resultTitle:'你的配色結果', export:'匯出配方', stale:'目標已變更，請重新搜尋以更新配方。',
  emptyTitle:'從一個顏色，開始配方設計。', emptyText:'選擇實測樣本或輸入 Lab 數值，即可搜尋對應的染料配方。',
  predictedColor:'模型預測色', predictedDelta:'預測色差 ΔE₇₆', previewNote:'D65／sRGB 螢幕色彩預覽。<br>色差以 Lab 數值計算。',
  candidatesEyebrow:'探索不同染料組合', candidates:'候選配方', fineTune:'微調這組配方', fineTuneText:'調整染料濃度，觀察模型預測色的變化。',
  reset:'重設配方', dyeColumn:'染料', concentrationColumn:'濃度調整', valueColumn:'濃度值', concentrationNote:'濃度沿用原始資料的數值尺度。',
  total:'總濃度', revert:'回到最佳化配方', methodEyebrow:'配色背後的技術', methodTitle:'把染整知識，轉化為逆向配方設計。',
  methodIntro:'先讓神經網路學會「配方如何形成顏色」，再反向搜尋能接近目標色的染料濃度。',
  method1:'從製程資料出發', method1text:'整合實測 Lab、染料濃度與布料種類，並從歷史配方取得適合的染料組合。',
  method2:'學習配方與色彩的關係', method2text:'10 層神經網路將 44 個染料濃度欄位與 3 個布種編碼，轉換為 L*、a*、b*。',
  method3:'從目標色反推濃度', method3text:'固定已訓練的模型權重，最佳化染料濃度，再依模型預測色差排列候選配方。',
  technical:'模型細節與工程設計', architecture:'模型架構', activation:'啟動函數', parameters:'訓練參數', runtime:'互動執行環境',
  engineeringText:'新版沿用原始模型權重，搭配可重現、有濃度範圍限制的逆向搜尋。Web Worker 在背景運算，保持操作流暢；數值測試比對瀏覽器預測與原始 HDF5 權重。',
  readDocs:'閱讀完整專案說明', footer:'紡織配色預測專案 · 原始開發 2018–2019 · 介面更新 2026'
};
const messages = {
  en:{ready:'Original model ready',searching:'Optimizing dye combinations',complete:'Recipe search complete',error:'Unable to load the model. Check your connection and reload.',invalid:'Enter finite Lab values: L* 0–100; a* and b* −128–128.',failed:'The search could not finish. Please try again.',recipe:'Recipe',options:'candidates',ranked:'Ranked by model-predicted ΔE₇₆',edited:'Concentrations edited · live model prediction',dye:'Dye',concentration:'concentration',loadSample:'Load sample',none:'No recipes for this fabric.',initial:'Searching the first sample…'},
  zh:{ready:'原始模型已就緒',searching:'正在最佳化染料組合',complete:'配方搜尋完成',error:'模型載入失敗，請確認連線後重新載入。',invalid:'請輸入有效的 Lab 數值：L* 0–100；a*、b* −128–128。',failed:'搜尋未完成，請重新嘗試。',recipe:'配方',options:'組候選',ranked:'依模型預測色差 ΔE₇₆ 排序',edited:'濃度已調整 · 即時模型預測',dye:'染料',concentration:'濃度',loadSample:'載入樣本',none:'此布種沒有可用配方。',initial:'正在搜尋第一組樣本…'}
};
const presets = [
  {en:'Seafoam',zh:'海沫綠',lab:[58.28,-29.09,-7.73],fabric:0},
  {en:'Blue iris',zh:'鳶尾藍',lab:[47.36,3.73,-19.86],fabric:1},
  {en:'Coral',zh:'珊瑚橘',lab:[72.22,36.12,24.07],fabric:1},
  {en:'Mulberry',zh:'桑椹紫',lab:[27.43,23.27,-6.5],fabric:1}
];
let language='en', decoder, metadata, worker, job=0, resultSet=null, selected=0, current=[], busy=false, edited=false, state='loading', progress='';
const msg = k => messages[language][k] || k;
const format = n => n < 0.01 ? '<0.01' : n.toFixed(2);
const readTarget = () => ['target-l','target-a','target-b'].map(id => $(id).value.trim()==='' ? NaN : Number($(id).value));
const fabricIndex = () => Number($('fabric').value);
const labText = lab => `L* ${lab[0].toFixed(2)}  a* ${lab[1].toFixed(2)}  b* ${lab[2].toFixed(2)}`;

function paintSwatch(el,lab) {
  el.style.backgroundColor=rgbCss(lab);
  const rgb=labToRgb(lab), luminance=rgb.reduce((s,v,i)=>s+[.2126,.7152,.0722][i]*(v/255<=.04045?v/255/12.92:((v/255+.055)/1.055)**2.4),0);
  el.style.color=luminance>.32?'#222431':'#fff';
}
function targetChanged() {
  const target=readTarget();
  if(target.every(Number.isFinite)) {
    paintSwatch($('target-swatch'),target);
    $('target-hex').textContent='#'+labToRgb(target).map(v=>v.toString(16).padStart(2,'0')).join('').toUpperCase();
  }
  const stale=resultSet && (fabricIndex()!==resultSet.fabric || target.some((v,i)=>v!==resultSet.target[i]));
  $('stale-notice').hidden=!stale;
  $('export').disabled=!resultSet || stale || busy;
  $('input-error').hidden=true;
}
function renderPresets() {
  $('presets').replaceChildren(...presets.map(p=>{
    const button=document.createElement('button'); button.type='button';
    button.setAttribute('aria-label',`${msg('loadSample')}: ${p[language]}`);
    button.innerHTML=`<i style="background:${rgbCss(p.lab)}" aria-hidden="true"></i><span>${p[language]}</span>`;
    button.disabled=busy;
    button.addEventListener('click',()=>{
      ['target-l','target-a','target-b'].forEach((id,i)=>$(id).value=p.lab[i]);
      $('fabric').value=p.fabric; targetChanged(); if(decoder) startSearch();
    });return button;
  }));
}
function setBusy(value) {
  busy=value;
  $('target-form').querySelectorAll('input,select,button').forEach(el=>el.disabled=value);
  $('search').disabled=value||!decoder;
  $('export').disabled=value||!resultSet||!$('stale-notice').hidden;
  $('status-icon').className=`status-icon ${value?'busy':decoder?'ready':''}`;
}
function status() {
  $('status').textContent=(state==='loading'?(language==='en'?english.loading:chinese.loading):msg(state))+(progress?` · ${progress}`:'');
}
function setLanguage(lang) {
  language=lang; document.documentElement.lang=lang==='zh'?'zh-Hant':'en';
  document.querySelectorAll('[data-i18n]').forEach(el=>el.innerHTML=(lang==='zh'?chinese:english)[el.dataset.i18n]||english[el.dataset.i18n]);
  ['en','zh'].forEach(v=>{ $('language-'+v).classList.toggle('active',v===lang);$('language-'+v).setAttribute('aria-pressed',String(v===lang)); });
  $('candidate-list').setAttribute('aria-label',lang==='zh'?'候選配方':'Candidate recipes');
  renderPresets();status();
  if(!$('input-error').hidden) $('input-error').textContent=msg('invalid');
  if(resultSet){renderCandidates();renderControls();renderPrediction();}
}
function buildWorker() {
  worker?.terminate();worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});
  worker.onmessage=({data})=>{
    if(data.id!==job)return;
    if(data.type==='progress'){progress=`${data.done}/${data.total}`;status();}
    if(data.type==='result'){
      resultSet.results=data.results;selected=0;edited=false;current=[...data.results[0].concentrations];
      state='complete';progress='';setBusy(false);status();
      $('results').hidden=false;$('empty-state').hidden=true;$('stale-notice').hidden=true;
      renderCandidates();renderControls();renderPrediction();
    }
    if(data.type==='error') failSearch(data.message==='NO_CANDIDATES'?'none':'failed');
  };
  worker.onerror=()=>failSearch('failed');
}
function failSearch(key) {state=key;progress='';resultSet=null;setBusy(false);status();$('results').hidden=true;$('empty-state').hidden=false;buildWorker();}
function startSearch(event) {
  event?.preventDefault();if(busy||!decoder)return;
  const target=readTarget(),fabric=fabricIndex();
  try{validateTarget(target,fabric,metadata);}catch{
    $('input-error').textContent=msg('invalid');$('input-error').hidden=false;return;
  }
  resultSet={target:[...target],fabric,results:[]};$('results').hidden=true;$('empty-state').hidden=false;$('stale-notice').hidden=true;
  $('input-error').hidden=true;state='searching';progress='0/8';setBusy(true);status();
  worker.postMessage({id:++job,target,fabric});
}
function renderCandidates(){
  if(!resultSet?.results.length)return;
  $('candidate-count').textContent=`${resultSet.results.length} ${msg('options')}`;
  $('candidate-list').replaceChildren(...resultSet.results.map((r,i)=>{
    const b=document.createElement('button');b.type='button';b.className='candidate'+(i===selected?' active':'');b.setAttribute('aria-pressed',String(i===selected));
    b.innerHTML=`<span>${msg('recipe')} ${String(i+1).padStart(2,'0')}</span><strong>ΔE ${format(r.delta)}</strong><em>${r.dyes.map(d=>metadata.dyes[d]).join(' · ')}</em>`;
    b.addEventListener('click',()=>{selected=i;current=[...r.concentrations];edited=false;renderCandidates();renderControls();renderPrediction();});return b;
  }));
}
function renderControls(){
  if(!resultSet?.results.length)return;
  const r=resultSet.results[selected];
  $('recipe-controls').replaceChildren(...r.dyes.map((d,i)=>{
    const row=document.createElement('div');row.className='dye-row';
    const hue=metadata.dye_swatches[String(d)];
    row.innerHTML=`<span class="dye-label"><i class="dye-dot" style="background:${hue?rgbCss(hue):'#a59bb7'}" aria-hidden="true"></i>${metadata.dyes[d]}</span><input type="range" min="0" max="5" step="0.0001" value="${current[i]}" aria-label="${msg('dye')} ${metadata.dyes[d]} ${msg('concentration')}"><input type="number" min="0" max="5" step="0.0001" value="${current[i].toFixed(4)}" aria-label="${msg('dye')} ${metadata.dyes[d]} ${msg('concentration')} ${language==='en'?'value':'數值'}">`;
    const [slider,number]=row.querySelectorAll('input');
    const update=(source,other)=>{
      if(source.value.trim()===''||!Number.isFinite(Number(source.value))||Number(source.value)<0||Number(source.value)>5){source.setAttribute('aria-invalid','true');$('export').disabled=true;return;}
      source.removeAttribute('aria-invalid');other.removeAttribute('aria-invalid');current[i]=Number(source.value);other.value=source===slider?current[i].toFixed(4):current[i];
      edited=true;renderPrediction();targetChanged();
      if($('recipe-controls').querySelector('[aria-invalid="true"]'))$('export').disabled=true;
    };
    slider.addEventListener('input',()=>update(slider,number));number.addEventListener('input',()=>update(number,slider));
    return row;
  }));
}
function renderPrediction(){
  if(!resultSet?.results.length)return;
  const r=resultSet.results[selected], lab=decoder.predict(r.dyes,current,resultSet.fabric), target=resultSet.target;
  paintSwatch($('reference-color'),target);paintSwatch($('predicted-color'),lab);
  $('reference-values').textContent=labText(target);$('predicted-values').textContent=labText(lab);
  $('delta').textContent=format(deltaE(target,lab));
  $('residuals').innerHTML=['ΔL*','Δa*','Δb*'].map((name,i)=>`<div><span>${name}</span><strong>${lab[i]-target[i]>=0?'+':''}${(lab[i]-target[i]).toFixed(2)}</strong></div>`).join('');
  $('total').textContent=current.reduce((s,v)=>s+v,0).toFixed(4);
  $('result-caption').textContent=`${metadata.fabrics[resultSet.fabric]} · ${msg(edited?'edited':'ranked')}`;
  $('revert').hidden=!edited;
}
function resetRecipe(){if(!resultSet?.results.length)return;current=[...resultSet.results[selected].concentrations];edited=false;renderControls();renderPrediction();targetChanged();}
function exportRecipe(){
  if(!resultSet?.results.length||$('export').disabled)return;
  const r=resultSet.results[selected],lab=decoder.predict(r.dyes,current,resultSet.fabric);
  const rows=[['fabric','dye','concentration','target_L','target_a','target_b','predicted_L','predicted_a','predicted_b','predicted_deltaE76','edited']];
  r.dyes.forEach((d,i)=>rows.push([metadata.fabrics[resultSet.fabric],metadata.dyes[d],current[i].toFixed(8),...resultSet.target,...lab.map(v=>v.toFixed(8)),deltaE(lab,resultSet.target).toFixed(8),edited]));
  const csv=rows.map(row=>row.join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download=`PigPredict-${metadata.fabrics[resultSet.fabric]}-recipe-${selected+1}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

$('language-en').addEventListener('click',()=>setLanguage('en'));$('language-zh').addEventListener('click',()=>setLanguage('zh'));
$('target-form').addEventListener('submit',startSearch);
['target-l','target-a','target-b','fabric'].forEach(id=>$(id).addEventListener('input',targetChanged));
$('reset-recipe').addEventListener('click',resetRecipe);$('revert').addEventListener('click',resetRecipe);$('export').addEventListener('click',exportRecipe);
setLanguage('en');targetChanged();
try{
  const [m,b]=await Promise.all(['model.json','weights.bin'].map(p=>fetch(new URL(`data/${p}`,import.meta.url))));
  if(!m.ok||!b.ok)throw new Error('MODEL_LOAD_FAILED');
  metadata=await m.json();decoder=new Decoder(metadata,await b.arrayBuffer());buildWorker();state='ready';setBusy(false);status();startSearch();
}catch{state='error';status();$('search').disabled=true;}
