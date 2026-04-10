/* ═══════════════════
   CONTRIBUTOR MODE
═══════════════════ */

/* ─── UX phase management ─── */

function contribShowPhase(phase) {
  ['idle', 'rec', 'review', 'label', 'done'].forEach(p => {
    const el = document.getElementById('contrib-p-' + p);
    if (el) el.style.display = p === phase ? 'flex' : 'none';
  });
}

/* ─── Entry: user clicks "Start Contributing" ─── */

function contribOpenConsent() {
  document.getElementById('contrib-modal').classList.add('show');
}

function contribConsentDecline() {
  document.getElementById('contrib-modal').classList.remove('show');
}

async function contribConsentAccept() {
  document.getElementById('contrib-modal').classList.remove('show');
  contribSessionId = _contribUUID();

  try {
    const res = await fetch(`${SERVER}/contributor/consent`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: contribSessionId,
        user_agent: navigator.userAgent,
        collection_mode: 'both',
        storage_backend: 'local',
      }),
    });
    const data = await res.json();
    contribConsentId = data.consent_id;
  } catch(e) {
    setBadge('contrib-badge', 'err', 'no server');
    return;
  }

  await _contribBeginSession();
}

/* ─── Session lifecycle ─── */

async function _contribBeginSession() {
  const myId = ++_contribStartId;
  document.getElementById('contrib-start-btn').disabled = true;

  try {
    if (!streamC) {
      const newStream = await navigator.mediaDevices.getUserMedia({ video: { width: W, height: H } });
      if (_contribStartId !== myId) { newStream.getTracks().forEach(t => t.stop()); return; }
      streamC = newStream;
      const vid = document.getElementById('webcam-contrib');
      vid.srcObject = streamC;
      vid.style.display = 'block';
      document.getElementById('contrib-idle-cam').style.display = 'none';
      const kp = document.getElementById('kp-contrib');
      kp.width = W; kp.height = H;
      if (!mpC) mpC = makeMP(onContribMP);
      rafC = startRaf(vid, mpC);
    }

    const startRes = await fetch(`${SERVER}/contributor/start`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: contribSessionId,
        consent_id: contribConsentId,
        collection_mode: 'both',
      }),
    });
    const startData = await startRes.json();
    contribVideoId = startData.video_id;

    contribShowPhase('rec');
    document.getElementById('contrib-stop-hdr').style.display = '';
    setBadge('contrib-badge', '', 'get ready');
    document.getElementById('contrib-rec-status').textContent = 'get ready to shuffle';
    document.getElementById('contrib-ring-num').textContent = GAME_SECS;
    document.getElementById('contrib-ring-track').style.strokeDashoffset = '0';
    document.getElementById('contrib-ring-track').style.stroke = 'var(--ok)';
    document.getElementById('contrib-ring-num').style.color = 'var(--ok)';

    await countdown('contrib-cd', 'contrib-cd-n', 3);
    if (_contribStartId !== myId) return;

    _contribBeginRecording(myId);

  } catch(e) {
    if (_contribStartId !== myId) return;
    setBadge('contrib-badge', 'err', 'no camera');
    document.getElementById('contrib-start-btn').disabled = false;
    console.error(e);
  }
}

function _contribBeginRecording(myId) {
  contribRecording = true;
  contribElapsed = 0;
  contribKpHistory = [];
  contribMaskHistory = [];
  document.getElementById('contrib-prog-wrap').classList.add('show');
  document.getElementById('contrib-prog').style.width = '0%';
  setBadge('contrib-badge', 'contrib', 'recording');
  document.getElementById('contrib-rec-status').textContent = 'recording shuffle\u2026';

  contribInterval = setInterval(contribSendFrame, INTERVAL);
  contribTimer = setInterval(() => {
    contribElapsed++;
    const rem  = GAME_SECS - contribElapsed;
    const prog = contribElapsed / GAME_SECS;
    document.getElementById('contrib-ring-num').textContent = rem > 0 ? rem : '0';
    document.getElementById('contrib-ring-track').style.strokeDashoffset = CIRC * prog;
    document.getElementById('contrib-prog').style.width = (prog * 100) + '%';
    document.getElementById('contrib-rec-status').textContent =
      rem > 0 ? `recording \u2014 ${rem}s remaining` : 'processing\u2026';
    if (contribElapsed >= GAME_SECS) {
      clearInterval(contribTimer); contribTimer = null;
      _contribFinishRecording();
    }
  }, 1000);
}

async function _contribFinishRecording() {
  clearInterval(contribInterval); contribInterval = null;
  contribRecording = false;
  document.getElementById('contrib-prog-wrap').classList.remove('show');
  document.getElementById('contrib-ring-track').style.stroke = 'var(--ok)';
  document.getElementById('contrib-ring-num').textContent = '\u2713';
  document.getElementById('contrib-ring-num').style.color = 'var(--ok)';
  setBadge('contrib-badge', '', 'reviewing');

  try {
    await fetch(`${SERVER}/contributor/stop`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: contribSessionId }),
    });
  } catch(e) {}

  contribShowPhase('review');
  _contribStartSkeletonReplay();
}

/* ─── Skeleton preview animation ─── */

let _contribReplayId = 0;

function _contribStartSkeletonReplay() {
  const myReplay = ++_contribReplayId;
  const canvas = document.getElementById('contrib-skeleton-canvas');
  if (!canvas) return;

  // Size canvas to match its displayed size
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width  = rect.width  || 220;
  canvas.height = rect.height || 120;

  const ctx = canvas.getContext('2d');
  const kpData = contribKpHistory.slice(0, contribKpHistory.length);
  const mkData = contribMaskHistory.slice(0, contribMaskHistory.length);
  const total  = kpData.length;
  if (total === 0) return;

  let frame = 0;
  function step() {
    if (_contribReplayId !== myReplay) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (frame >= total) { frame = 0; }

    const kp = kpData[frame];
    const mk = mkData[frame];
    for (let h = 0; h < 2; h++) {
      if (!mk[h]) continue;
      const color = h === 0 ? '#38bdf8' : '#fb923c';
      kp[h].forEach((lm, j) => {
        const x = lm[0] * canvas.width;
        const y = lm[1] * canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, j === 0 ? 4 : 2.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = j === 0 ? 1 : 0.7;
        ctx.fill();
        ctx.globalAlpha = 1;
      });
    }
    frame++;
    setTimeout(() => requestAnimationFrame(step), 80);
  }
  requestAnimationFrame(step);
}

function contribReplay() {
  _contribStartSkeletonReplay();
}

/* ─── Labeling ─── */

let _contribStartSel = null, _contribEndSel = null;

function contribGoToLabel() {
  _contribStartSel = null;
  _contribEndSel   = null;
  ['start-L', 'start-R', 'end-L', 'end-R'].forEach(id => {
    const el = document.getElementById('chl-' + id);
    if (el) el.classList.remove('sel');
  });
  document.getElementById('contrib-label-err').classList.remove('show');
  contribShowPhase('label');
  setBadge('contrib-badge', '', 'labeling');
}

function contribSelectHand(which, hand) {
  const side = hand === 'left' ? 'L' : 'R';
  const other = hand === 'left' ? 'R' : 'L';
  document.getElementById(`chl-${which}-${side}`).classList.add('sel');
  document.getElementById(`chl-${which}-${other}`).classList.remove('sel');
  if (which === 'start') _contribStartSel = hand;
  else                   _contribEndSel   = hand;
}

async function contribSubmitLabel() {
  if (!_contribStartSel || !_contribEndSel) {
    document.getElementById('contrib-label-err').classList.add('show');
    return;
  }
  document.getElementById('contrib-label-err').classList.remove('show');
  document.getElementById('contrib-submit-btn').disabled = true;

  try {
    const res = await fetch(`${SERVER}/contributor/label`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: contribSessionId,
        start_hand: _contribStartSel,
        end_hand:   _contribEndSel,
      }),
    });
    const data = await res.json();
    document.getElementById('contrib-done-vid').textContent = `Video ID: ${data.video_id}`;
    contribShowPhase('done');
    setBadge('contrib-badge', 'ok', 'submitted');
  } catch(e) {
    setBadge('contrib-badge', 'err', 'save failed');
    document.getElementById('contrib-submit-btn').disabled = false;
  }
}

/* ─── Frame sender ─── */

async function contribSendFrame() {
  if (!contribRecording) { clearInterval(contribInterval); return; }

  // Build keypoints array from latest MP result (corrected for MP mirror flip)
  const kp   = [Array(21).fill(null).map(() => [0, 0, 0]),
                 Array(21).fill(null).map(() => [0, 0, 0])];
  const mask = [false, false];
  if (contribLatestMP && contribLatestMP.multiHandLandmarks) {
    contribLatestMP.multiHandLandmarks.forEach((lms, i) => {
      const isLeft = contribLatestMP.multiHandedness[i].label === 'Right'; // MP mirror
      const hi = isLeft ? 0 : 1;
      kp[hi]   = lms.map(lm => [lm.x, lm.y, lm.z]);
      mask[hi] = true;
    });
  }

  // Store client-side for skeleton replay
  contribKpHistory.push(kp.map(h => h.map(lm => [...lm])));
  contribMaskHistory.push([...mask]);

  const frameB64 = captureJpeg(document.getElementById('webcam-contrib'));
  try {
    await fetch(`${SERVER}/contributor/frame`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id:  contribSessionId,
        frame:       frameB64,
        keypoints:   kp,
        mask:        mask,
        frame_index: contribKpHistory.length - 1,
      }),
    });
  } catch(e) {}
}

/* ─── MediaPipe callback ─── */

function onContribMP(res) {
  const canvas = document.getElementById('kp-contrib');
  const { detected, count } = drawHands(canvas, res);
  contribHandsIn    = detected;
  contribHandCount  = count;
  contribLatestMP   = res;

  const ltag = document.getElementById('contrib-ltag');
  const rtag = document.getElementById('contrib-rtag');
  ltag.className = 'ctag'; rtag.className = 'ctag';
  if (detected && res.multiHandLandmarks) {
    res.multiHandLandmarks.forEach((_, i) => {
      const isLeft = res.multiHandedness[i].label === 'Right';
      if (isLeft) ltag.classList.add('hand-left'); else rtag.classList.add('hand-right');
    });
  }
  document.getElementById('contrib-nhtag').classList.toggle('show', contribRecording && !detected);
}

/* ─── Stop / cleanup ─── */

function contribStop() {
  _contribStartId++;
  contribRecording = false;
  clearInterval(contribTimer);    contribTimer    = null;
  clearInterval(contribInterval); contribInterval = null;
  if (rafC)    { rafC();    rafC    = null; }
  if (streamC) { streamC.getTracks().forEach(t => t.stop()); streamC = null; }
  mpC = null;

  const vid = document.getElementById('webcam-contrib');
  if (vid) vid.style.display = 'none';
  const idle = document.getElementById('contrib-idle-cam');
  if (idle) idle.style.display = 'flex';
  const kp = document.getElementById('kp-contrib');
  if (kp) kp.getContext('2d').clearRect(0, 0, W, H);

  document.getElementById('contrib-stop-hdr').style.display = 'none';
  document.getElementById('contrib-prog-wrap').classList.remove('show');
  document.getElementById('contrib-nhtag').classList.remove('show');
  const btn = document.getElementById('contrib-start-btn');
  if (btn) { btn.textContent = 'Start Contributing'; btn.disabled = false; }
  contribShowPhase('idle');
  setBadge('contrib-badge', '', 'idle');
  _contribReplayId++;
}

function contribPlayAgain() {
  contribStop();
  // Immediately re-open consent for a new session
  contribOpenConsent();
}

/* ─── Helpers ─── */

function _contribUUID() {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  return 'xxxx-xxxx-4xxx-yxxx-xxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}
