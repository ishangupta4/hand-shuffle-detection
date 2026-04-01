/* ═══════════════════
   GAME MODE
═══════════════════ */

async function gameStart() {
  const myId = ++_gameStartId;
  document.getElementById('game-start-btn').disabled = true;
  document.getElementById('game-stop-btn').disabled  = false;
  document.getElementById('game-stop-hdr').style.display = '';
  try {
    if (!streamG) {
      const newStream = await navigator.mediaDevices.getUserMedia({ video: { width: W, height: H } });
      // Guard: gameStop() may have run while getUserMedia was pending
      if (_gameStartId !== myId) {
        newStream.getTracks().forEach(t => t.stop());
        return;
      }
      streamG = newStream;
      const vid = document.getElementById('webcam-game');
      vid.srcObject = streamG;
      vid.style.display = 'block';
      document.getElementById('game-idle').style.display = 'none';
      const kp = document.getElementById('kp-game');
      kp.width = W; kp.height = H;
      if (!mpG) mpG = makeMP(onGameMP);
      rafG = startRaf(vid, mpG);
    }
    serverReset(); // fire-and-forget — completes well within the 3s countdown
    gameResetUI();
    await countdown('game-cd', 'game-cd-n', 3);
    // Guard: gameStop() may have run during the countdown
    if (_gameStartId !== myId) return;
    gameBeginRecording();
  } catch(e) {
    if (_gameStartId !== myId) return; // gameStop() already cleaned up
    setBadge('game-badge', 'err', 'no camera');
    document.getElementById('game-start-btn').disabled = false;
    document.getElementById('game-stop-btn').disabled  = true;
    document.getElementById('game-stop-hdr').style.display = 'none';
    console.error(e);
  }
}

/* Single authoritative function that starts the 15s recording.
   Called by both gameStart (first round) and newGameRound (subsequent rounds). */
function gameBeginRecording() {
  gameActive   = true;
  gameElapsed  = 0;
  gameFinished = false;
  document.getElementById('result-card').className = 'result-card';
  setBadge('game-badge', 'game', 'recording');
  document.getElementById('game-state-lbl').textContent = 'recording shuffle…';
  document.getElementById('game-ptag').className    = 'ctag rec-tag';
  document.getElementById('game-ptag').textContent  = 'REC';
  document.getElementById('game-prog-wrap').classList.add('show');
  document.getElementById('game-prog').style.width  = '0%';
  document.getElementById('game-start-btn').disabled = true;
  document.getElementById('game-stop-btn').disabled  = false;

  gameInterval = setInterval(gameSendFrame, INTERVAL);
  gameTimer = setInterval(() => {
    gameElapsed++;
    const rem  = GAME_SECS - gameElapsed;
    const prog = gameElapsed / GAME_SECS;
    document.getElementById('ring-num').textContent = rem;
    document.getElementById('ring-track').style.strokeDashoffset = CIRC * prog;
    document.getElementById('game-prog').style.width = (prog * 100) + '%';
    document.getElementById('game-state-lbl').textContent =
      rem > 0 ? `recording — ${rem}s remaining` : 'analysing…';
    if (gameElapsed >= GAME_SECS) { clearInterval(gameTimer); gameTimer = null; gameFinish(); }
  }, 1000);
}

async function gameFinish() {
  clearInterval(gameInterval); gameInterval = null;
  gameActive   = false;
  gameFinished = true;
  document.getElementById('game-prog-wrap').classList.remove('show');
  document.getElementById('ring-track').style.strokeDashoffset = CIRC;
  document.getElementById('ring-track').style.stroke = 'var(--ok)';
  document.getElementById('ring-num').textContent = '✓';
  document.getElementById('ring-num').style.color  = 'var(--ok)';
  document.getElementById('game-ptag').className    = 'ctag';
  document.getElementById('game-ptag').textContent  = '—';
  setBadge('game-badge', '', 'result');
  document.getElementById('game-state-lbl').textContent = 'analysing 15s window…';

  try {
    const res = await fetch(`${SERVER}/predict`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: captureJpeg(document.getElementById('webcam-game')) }),
    });
    const data = await res.json();
    if (data.ready) showGameResult(data);
    else document.getElementById('game-state-lbl').textContent = 'not enough data — try again';
  } catch(e) {
    document.getElementById('game-state-lbl').textContent = 'server error';
  }

  // Only offer New Round if the user hasn't already pressed Stop
  if (!streamG) return;
  document.getElementById('game-start-btn').textContent = 'New Round';
  document.getElementById('game-start-btn').onclick     = newGameRound;
  document.getElementById('game-start-btn').disabled    = false;
  document.getElementById('game-stop-btn').disabled     = true;
}

function showGameResult(data) {
  const pl = (data.prob_left  * 100).toFixed(1);
  const pr = (data.prob_right * 100).toFixed(1);
  const P  = data.prediction === 'left' ? 'L' : 'R';
  document.getElementById('g-fL').style.width  = pl + '%';
  document.getElementById('g-fR').style.width  = pr + '%';
  document.getElementById('g-cvL').textContent = pl + '%';
  document.getElementById('g-cvR').textContent = pr + '%';
  const rc = document.getElementById('result-card');
  rc.className = 'result-card show ' + P;
  document.getElementById('res-hand').className   = 'res-hand ' + P;
  document.getElementById('res-hand').textContent = data.prediction.toUpperCase();
  document.getElementById('res-conf').textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;
  document.getElementById('game-state-lbl').textContent = 'prediction complete';
  setBadge('game-badge', 'live', 'done');
  document.getElementById('game-ptag').className    = 'ctag pred-' + data.prediction;
  document.getElementById('game-ptag').textContent  = data.prediction.toUpperCase() + ' HAND';
}

async function newGameRound() {
  const myId = ++_gameStartId;
  document.getElementById('game-start-btn').disabled = true;
  serverReset(); // fire-and-forget — completes well within the 3s countdown
  gameResetUI();
  await countdown('game-cd', 'game-cd-n', 3);
  // Guard: gameStop() may have run during the countdown
  if (_gameStartId !== myId) return;
  gameBeginRecording();
}

function gameStop() {
  _gameStartId++; // invalidate any in-flight gameStart / newGameRound
  gameActive = false;
  clearInterval(gameTimer);    gameTimer    = null;
  clearInterval(gameInterval); gameInterval = null;
  if (rafG)    { rafG();    rafG    = null; }
  if (streamG) { streamG.getTracks().forEach(t => t.stop()); streamG = null; }
  mpG = null;
  const vid = document.getElementById('webcam-game');
  vid.style.display = 'none';
  document.getElementById('kp-game').getContext('2d').clearRect(0, 0, W, H);
  document.getElementById('game-idle').style.display = 'flex';
  document.getElementById('game-prog-wrap').classList.remove('show');
  document.getElementById('game-stop-hdr').style.display = 'none';
  document.getElementById('game-stop-btn').disabled  = true;
  document.getElementById('game-nhtag').classList.remove('show');
  const btn = document.getElementById('game-start-btn');
  btn.textContent = 'Start Round'; btn.onclick = gameStart; btn.disabled = false;
  setBadge('game-badge', '', 'idle');
  gameResetUI();
  setInfoBox('game-info-inner', 'game-info-empty', []);
}

function onGameMP(res) {
  const canvas = document.getElementById('kp-game');
  const { detected, count, quality } = drawHands(canvas, res);
  gameHandsIn = detected; gameHandCount = count; gameQuality = quality;

  const ltag = document.getElementById('game-ltag');
  const rtag = document.getElementById('game-rtag');
  ltag.className = 'ctag'; rtag.className = 'ctag';
  if (detected) {
    res.multiHandLandmarks.forEach((_, i) => {
      const isLeft = res.multiHandedness[i].label === 'Right';
      if (isLeft) ltag.classList.add('hand-left'); else rtag.classList.add('hand-right');
    });
  }
  document.getElementById('game-nhtag').classList.toggle('show', gameActive && !detected);
  if (gameActive) setInfoBox('game-info-inner', 'game-info-empty', computeWarnings(detected, count, quality, gameElapsed * Math.round(1000 / INTERVAL), gameActive));
}

async function gameSendFrame() {
  if (!gameActive) { clearInterval(gameInterval); return; }
  if (!gameHandsIn) { document.getElementById('game-buftag').textContent = 'no hands — paused'; return; }
  try {
    const res = await fetch(`${SERVER}/predict`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: captureJpeg(document.getElementById('webcam-game')) }),
    });
    if (!gameActive) return;
    const data = await res.json();
    document.getElementById('game-buftag').textContent = `${data.num_frames_buffered} frames`;
  } catch(e) {}
}

function gameResetUI() {
  document.getElementById('ring-num').textContent = GAME_SECS;
  document.getElementById('ring-num').style.color = 'var(--game)';
  document.getElementById('ring-track').style.strokeDashoffset = '0';
  document.getElementById('ring-track').style.stroke = 'var(--game)';
  document.getElementById('game-prog').style.width = '0%';
  document.getElementById('game-state-lbl').textContent = 'ready to start';
  document.getElementById('result-card').className = 'result-card';
  document.getElementById('game-ptag').className   = 'ctag';
  document.getElementById('game-ptag').textContent = '—';
  document.getElementById('game-buftag').textContent = '—';
}