/* ─── SHARED UI HELPERS ─── */

/* Status badge */
function setBadge(id, cls, text) {
  const el = document.getElementById(id);
  el.className = 'sys-badge ' + cls;
  el.textContent = text;
}

/* Info / warnings box */
function setInfoBox(innerId, emptyId, warnings) {
  const inner = document.getElementById(innerId);
  const sig = warnings.map(w => w.type + w.msg).join('\0');
  if (inner.dataset.sig === sig) return;
  inner.dataset.sig = sig;
  const empty = document.getElementById(emptyId);
  inner.querySelectorAll('.info-row').forEach(el => el.remove());
  if (!warnings.length) { empty.style.display = 'block'; return; }
  empty.style.display = 'none';
  warnings.forEach(w => {
    const row = document.createElement('div');
    row.className = 'info-row';
    row.innerHTML = `<div class="info-dot ${w.type}"></div><span class="info-msg">${w.msg}</span>`;
    inner.appendChild(row);
  });
}

function computeWarnings(detected, count, quality, bufFrames, active) {
  const w = [];
  if (!active) return w;
  if (!detected) { w.push({ type: 'warn', msg: 'No hands detected — show both hands to the camera' }); return w; }
  if (count === 1) w.push({ type: 'warn', msg: 'Only one hand visible — both hands needed for accurate prediction' });
  if (quality < 0.5)       w.push({ type: 'err',  msg: 'Hand feature quality low — improve lighting or move closer' });
  else if (quality < 0.85) w.push({ type: 'warn', msg: 'Some landmarks near frame edge — keep hands fully visible' });
  if (bufFrames > 0 && bufFrames < MIN_FRAMES) w.push({ type: 'warn', msg: 'Collecting frames — keep hands visible' });
  if (!w.length && count === 2 && quality >= 0.85) w.push({ type: 'ok', msg: 'Both hands detected — good signal quality' });
  return w;
}

/* Countdown overlay */
function countdown(wrapId, numId, n) {
  return new Promise(resolve => {
    const wrap  = document.getElementById(wrapId);
    const numEl = document.getElementById(numId);
    numEl.textContent = n;
    wrap.classList.add('show');
    const t = setInterval(() => {
      n--;
      if (n <= 0) { clearInterval(t); wrap.classList.remove('show'); resolve(); }
      else {
        numEl.textContent = n;
        numEl.style.animation = 'none';
        void numEl.offsetWidth;
        numEl.style.animation = '';
      }
    }, 1000);
  });
}

/* Frame capture */
const _capCanvas = document.createElement('canvas');
_capCanvas.width  = W;
_capCanvas.height = H;
const _capCtx = _capCanvas.getContext('2d');

function captureJpeg(videoEl) {
  _capCtx.drawImage(videoEl, 0, 0, W, H);
  return _capCanvas.toDataURL('image/jpeg', .72).split(',')[1];
}

/* Server helpers */
async function serverReset() {
  try { await fetch(`${SERVER}/reset`, { method: 'POST' }); } catch(e) {}
}