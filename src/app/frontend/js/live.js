/* ═══════════════════
   LIVE MODE
═══════════════════ */

async function liveStart() {
  const myId = ++_liveStartId;
  const btn = document.getElementById('live-start-btn');
  btn.disabled = true;
  try {
    if (!streamL) {
      const newStream = await navigator.mediaDevices.getUserMedia({ video: { width: W, height: H } });
      // Guard: liveStop() may have run while getUserMedia was pending
      if (_liveStartId !== myId) {
        newStream.getTracks().forEach(t => t.stop());
        return;
      }
      streamL = newStream;
      const vid = document.getElementById('webcam-live');
      vid.srcObject = streamL;
      vid.style.display = 'block';
      document.getElementById('live-idle').style.display = 'none';
      const kp = document.getElementById('kp-live');
      kp.width = W; kp.height = H;
      if (!mpL) mpL = makeMP(onLiveMP);
      rafL = startRaf(vid, mpL);
    }
    serverReset(); // fire-and-forget — completes well within the 3s countdown
    liveFreezeUI();
    await countdown('live-cd', 'live-cd-n', 3);
    // Guard: liveStop() may have run during the countdown
    if (_liveStartId !== myId) return;
    liveActive = true;
    document.getElementById('live-stop-btn').style.display = '';
    document.getElementById('live-reset-btn').disabled = false;
    setBadge('live-badge', 'live', 'live');
    liveSetWaiting('show your hands…');
    btn.textContent = 'Restart'; btn.disabled = false; btn.onclick = liveRestart;
    liveTimer = setInterval(liveSendFrame, INTERVAL);
  } catch(e) {
    if (_liveStartId !== myId) return; // liveStop() already cleaned up
    setBadge('live-badge', 'err', 'no camera');
    btn.disabled = false;
    console.error(e);
  }
}

async function liveRestart() {
  const myId = ++_liveStartId;
  const btn = document.getElementById('live-start-btn');
  btn.disabled = true; // prevent overlapping restarts from double-clicking
  clearInterval(liveTimer); liveTimer = null;
  liveActive = false;
  serverReset(); // fire-and-forget
  liveFreezeUI();
  await countdown('live-cd', 'live-cd-n', 3);
  // Guard: liveStop() may have run during the countdown
  if (_liveStartId !== myId) return;
  liveActive = true;
  setBadge('live-badge', 'live', 'live');
  liveSetWaiting('show your hands…');
  btn.disabled = false;
  liveTimer = setInterval(liveSendFrame, INTERVAL);
}

function liveStop() {
  _liveStartId++; // invalidate any in-flight liveStart / liveRestart
  liveActive = false;
  clearInterval(liveTimer); liveTimer = null;
  if (rafL) { rafL(); rafL = null; }
  if (streamL) { streamL.getTracks().forEach(t => t.stop()); streamL = null; }
  mpL = null;
  const vid = document.getElementById('webcam-live');
  vid.style.display = 'none';
  document.getElementById('kp-live').getContext('2d').clearRect(0, 0, W, H);
  document.getElementById('live-idle').style.display = 'flex';
  document.getElementById('live-stop-btn').style.display = 'none';
  document.getElementById('live-reset-btn').disabled = true;
  document.getElementById('live-nhtag').classList.remove('show');
  const btn = document.getElementById('live-start-btn');
  btn.textContent = 'Start Prediction'; btn.disabled = false; btn.onclick = liveStart;
  setBadge('live-badge', '', 'idle');
  liveFreezeUI();
  liveSetWaiting('start prediction to begin');
  setInfoBox('live-info-inner', 'live-info-empty', []);
}

async function liveReset() {
  const btn = document.getElementById('live-reset-btn');
  btn.disabled = true; // prevent multiple resets
  liveFreezeUI();
  liveSetWaiting('clearing buffer…');
  await serverReset();
  // Guard: liveStop() may have run during the await
  if (!liveActive) return;
  liveSetWaiting('buffer cleared — show hands');
  document.getElementById('live-buftag').textContent = '0 frames';
  btn.disabled = false;
}

function onLiveMP(res) {
  const canvas = document.getElementById('kp-live');
  const { detected, count, quality } = drawHands(canvas, res);
  liveHandsIn = detected; liveHandCount = count; liveQuality = quality;

  const ltag = document.getElementById('live-ltag');
  const rtag = document.getElementById('live-rtag');
  ltag.className = 'ctag'; rtag.className = 'ctag';
  if (detected) {
    res.multiHandLandmarks.forEach((_, i) => {
      const isLeft = res.multiHandedness[i].label === 'Right';
      if (isLeft) ltag.classList.add('hand-left'); else rtag.classList.add('hand-right');
    });
  }
  document.getElementById('live-nhtag').classList.toggle('show', liveActive && !detected);
}

async function liveSendFrame() {
  if (!liveActive) { clearInterval(liveTimer); return; }
  const bufs = parseInt(document.getElementById('live-buftag').textContent) || 0;
  setInfoBox('live-info-inner', 'live-info-empty', computeWarnings(liveHandsIn, liveHandCount, liveQuality, bufs, liveActive));
  if (!liveHandsIn) { document.getElementById('live-buftag').textContent = 'no hands — paused'; return; }
  try {
    const res = await fetch(`${SERVER}/predict`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: captureJpeg(document.getElementById('webcam-live')) }),
    });
    if (!liveActive) return;
    const data = await res.json();
    document.getElementById('live-buftag').textContent = `${data.num_frames_buffered} frames`;
    liveUpdateUI(data);
    setInfoBox('live-info-inner', 'live-info-empty', computeWarnings(liveHandsIn, liveHandCount, liveQuality, data.num_frames_buffered, liveActive));
  } catch(e) {
    if (liveActive) setBadge('live-badge', 'err', 'no server');
  }
}

function liveUpdateUI(data) {
  if (!data.ready) { liveSetWaiting(`collecting (${data.num_frames_buffered}/${MIN_FRAMES})…`); liveFreezeUI(); return; }
  const pl = (data.prob_left  * 100).toFixed(1);
  const pr = (data.prob_right * 100).toFixed(1);
  const P  = data.prediction === 'left' ? 'L' : 'R';
  document.getElementById('live-fL').style.width  = pl + '%';
  document.getElementById('live-fR').style.width  = pr + '%';
  document.getElementById('live-cvL').textContent = pl + '%';
  document.getElementById('live-cvR').textContent = pr + '%';
  document.getElementById('live-crL').className = 'conf-row ' + (P === 'L' ? '' : 'dim');
  document.getElementById('live-crR').className = 'conf-row ' + (P === 'R' ? '' : 'dim');
  document.getElementById('live-cvL').classList.toggle('on', P === 'L');
  document.getElementById('live-cvR').classList.toggle('on', P === 'R');
  document.getElementById('live-verdict').className = 'verdict-inner ' + P;
  document.getElementById('live-vhand').className   = 'v-hand ' + P;
  document.getElementById('live-vhand').textContent = data.prediction.toUpperCase();
  document.getElementById('live-vconf').textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;
  document.getElementById('live-ptag').className    = 'ctag pred-' + data.prediction;
  document.getElementById('live-ptag').textContent  = data.prediction.toUpperCase() + ' HAND';
  setBadge('live-badge', 'live', 'live');
}

function liveFreezeUI() {
  document.getElementById('live-fL').style.width  = '50%';
  document.getElementById('live-fR').style.width  = '50%';
  document.getElementById('live-cvL').textContent = '—';
  document.getElementById('live-cvR').textContent = '—';
  document.getElementById('live-cvL').classList.remove('on');
  document.getElementById('live-cvR').classList.remove('on');
  document.getElementById('live-crL').className = 'conf-row frozen';
  document.getElementById('live-crR').className = 'conf-row frozen';
  document.getElementById('live-verdict').className = 'verdict-inner';
  document.getElementById('live-ptag').className    = 'ctag';
  document.getElementById('live-ptag').textContent  = '—';
}

function liveSetWaiting(msg) {
  document.getElementById('live-vhand').className   = 'v-hand pulse';
  document.getElementById('live-vhand').textContent = '?';
  document.getElementById('live-vconf').textContent = msg;
}