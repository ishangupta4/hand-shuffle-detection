/* ═══════════════════
   CONTRIBUTOR MODE
═══════════════════ */

/* ─── UX phase management (view-contrib only: idle, rec, done) ─── */

function contribShowPhase(phase) {
  ['idle', 'rec', 'done'].forEach(p => {
    const el = document.getElementById('contrib-p-' + p);
    if (el) el.style.display = p === phase ? 'flex' : 'none';
  });
}

/* ─── Entry: user clicks "Start Contributing" ─── */

function contribOpenConsent() {
  if (contribHasConsented) {
    _contribBeginSession();
    return;
  }
  document.getElementById('contrib-modal').classList.add('show');
}

function contribConsentDecline() {
  document.getElementById('contrib-modal').classList.remove('show');
}

async function contribConsentAccept() {
  document.getElementById('contrib-modal').classList.remove('show');
  contribHasConsented = true;
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
    if (!res.ok) throw new Error(`consent failed: ${res.status}`);
    const data = await res.json();
    contribConsentId = data.consent_id;
  } catch(e) {
    setBadge('contrib-badge', 'err', 'no server');
    contribHasConsented = false;
    return;
  }

  _contribBeginSession();
}

/* ─── Session lifecycle ─── */

const CONTRIB_CIRC = 263.9;  // 2π × 42 (contrib ring radius)

async function _contribBeginSession() {
  const myId = ++_contribStartId;
  document.getElementById('contrib-start-btn').disabled = true;

  contribShowPhase('rec');
  document.getElementById('contrib-stop-hdr').style.display = '';
  setBadge('contrib-badge', '', 'starting\u2026');
  document.getElementById('contrib-rec-status').textContent = 'starting camera\u2026';
  document.getElementById('contrib-ring-num').textContent = GAME_SECS;
  document.getElementById('contrib-ring-track').style.strokeDashoffset = '0';
  document.getElementById('contrib-ring-track').style.stroke = 'var(--ok)';
  document.getElementById('contrib-ring-num').style.color = 'var(--ok)';

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
    if (!startRes.ok) throw new Error(`start failed: ${startRes.status}`);
    const startData = await startRes.json();
    contribVideoId = startData.video_id;

    document.getElementById('contrib-rec-status').textContent = 'get ready to shuffle';
    setBadge('contrib-badge', '', 'get ready');

    await countdown('contrib-cd', 'contrib-cd-n', 3);
    if (_contribStartId !== myId) return;

    _contribBeginRecording(myId);

  } catch(e) {
    if (_contribStartId !== myId) return;
    contribShowPhase('idle');
    document.getElementById('contrib-stop-hdr').style.display = 'none';
    setBadge('contrib-badge', 'err', 'camera error');
    document.getElementById('contrib-start-btn').disabled = false;
    console.error('Contributor session error:', e);
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
    document.getElementById('contrib-ring-track').style.strokeDashoffset = CONTRIB_CIRC * prog;
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
  setBadge('contrib-badge', '', 'processing\u2026');
  document.getElementById('contrib-rec-status').textContent = 'building preview\u2026';

  // Stop server recording — server builds masked video during this call
  let previewUrl = null;
  try {
    const res = await fetch(`${SERVER}/contributor/stop`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: contribSessionId }),
    });
    const data = await res.json();
    previewUrl = data.preview_url || null;
  } catch(e) {}

  // Stop camera + RAF (no longer needed)
  if (rafC) { rafC(); rafC = null; }
  if (streamC) { streamC.getTracks().forEach(t => t.stop()); streamC = null; }
  mpC = null;

  const vid = document.getElementById('webcam-contrib');
  if (vid) vid.style.display = 'none';
  const idle = document.getElementById('contrib-idle-cam');
  if (idle) idle.style.display = 'flex';
  const kpCanvas = document.getElementById('kp-contrib');
  if (kpCanvas) kpCanvas.getContext('2d').clearRect(0, 0, kpCanvas.width, kpCanvas.height);
  document.getElementById('contrib-stop-hdr').style.display = 'none';
  document.getElementById('contrib-nhtag').classList.remove('show');

  // Reset review view
  const loading  = document.getElementById('contrib-review-loading');
  const playback = document.getElementById('contrib-playback');
  if (playback) { playback.src = ''; playback.style.display = 'none'; }
  if (loading)  loading.style.display = 'flex';

  document.getElementById('sel-start-hand').value = '';
  document.getElementById('sel-end-hand').value   = '';
  document.getElementById('contrib-label-err').classList.remove('show');
  document.getElementById('contrib-submit-btn').disabled = false;

  // Switch to review page
  switchToContribReview();
  setBadge('contrib-review-badge', '', 'review');

  // Load video
  if (previewUrl && playback) {
    playback.src = `${SERVER}${previewUrl}`;
    playback.oncanplay = () => {
      if (loading) loading.style.display = 'none';
      playback.style.display = 'block';
      playback.play().catch(() => {});
    };
    playback.onerror = () => {
      if (loading) loading.textContent = 'Preview unavailable';
    };
  } else if (loading) {
    loading.textContent = 'Preview unavailable';
  }
}

/* ─── Discard & Try Again ─── */

function contribDiscard() {
  // Tell server to delete preview file and drop session
  if (contribSessionId) {
    fetch(`${SERVER}/contributor/discard`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: contribSessionId }),
    }).catch(() => {});
  }

  // Clean up playback
  const playback = document.getElementById('contrib-playback');
  if (playback) { playback.pause(); playback.src = ''; playback.style.display = 'none'; }
  const loading = document.getElementById('contrib-review-loading');
  if (loading) { loading.style.display = 'flex'; loading.textContent = 'Processing video\u2026'; }

  // Fresh session ID so the next attempt doesn't collide with the discarded one
  contribSessionId = _contribUUID();

  // Return to contrib view idle
  document.getElementById('view-contrib-review').classList.remove('active');
  document.getElementById('view-contrib').classList.add('active');
  const btn = document.getElementById('contrib-start-btn');
  if (btn) { btn.textContent = 'Start Contributing'; btn.disabled = false; }
  contribShowPhase('idle');
  setBadge('contrib-badge', '', 'idle');
}

/* ─── Labeling (on view-contrib-review) ─── */

async function contribSubmitLabel() {
  const startHand = document.getElementById('sel-start-hand').value;
  const endHand   = document.getElementById('sel-end-hand').value;

  if (!startHand || !endHand) {
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
        start_hand: startHand,
        end_hand:   endHand,
      }),
    });
    const data = await res.json();

    // Clean up playback
    const playback = document.getElementById('contrib-playback');
    if (playback) { playback.pause(); playback.src = ''; playback.style.display = 'none'; }
    const loading = document.getElementById('contrib-review-loading');
    if (loading) { loading.style.display = 'flex'; loading.textContent = 'Processing video\u2026'; }

    // Go to done phase on view-contrib
    document.getElementById('view-contrib-review').classList.remove('active');
    document.getElementById('view-contrib').classList.add('active');
    document.getElementById('contrib-done-vid').textContent = `Video ID: ${data.video_id}`;
    contribShowPhase('done');
    setBadge('contrib-badge', 'ok', 'submitted');
  } catch(e) {
    setBadge('contrib-review-badge', 'err', 'save failed');
    document.getElementById('contrib-submit-btn').disabled = false;
  }
}

/* ─── Frame sender ─── */

async function contribSendFrame() {
  if (!contribRecording) { clearInterval(contribInterval); return; }

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

/* ─── Stop / cleanup (recording phase only) ─── */

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
}

function contribPlayAgain() {
  contribStop();
  contribSessionId = _contribUUID();
  _contribBeginSession();
}

/* ─── Helpers ─── */

function _contribUUID() {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  return 'xxxx-xxxx-4xxx-yxxx-xxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}
