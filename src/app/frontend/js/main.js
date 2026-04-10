/* ─── NAVIGATION ─── */

function switchFromSplashToGame() {
  document.getElementById('view-splash').classList.remove('active');
  document.getElementById('view-game').classList.add('active');
}

function switchFromSplashToLive() {
  document.getElementById('view-splash').classList.remove('active');
  document.getElementById('view-live').classList.add('active');
}

function switchToSplash() {
  if (gameActive) gameStop();
  if (liveActive) liveStop();
  if (contribRecording) contribStop();
  document.getElementById('view-game').classList.remove('active');
  document.getElementById('view-live').classList.remove('active');
  document.getElementById('view-contrib').classList.remove('active');
  document.getElementById('view-contrib-review').classList.remove('active');
  document.getElementById('view-splash').classList.add('active');
}

function switchToContribReview() {
  document.getElementById('view-contrib').classList.remove('active');
  document.getElementById('view-contrib-review').classList.add('active');
}

function switchToGame() {
  document.getElementById('view-live').classList.remove('active');
  document.getElementById('view-contrib').classList.remove('active');
  document.getElementById('view-game').classList.add('active');
}

function switchToLive() {
  if (gameActive) gameStop();
  document.getElementById('view-game').classList.remove('active');
  document.getElementById('view-contrib').classList.remove('active');
  document.getElementById('view-live').classList.add('active');
}

function switchFromSplashToContrib() {
  document.getElementById('view-splash').classList.remove('active');
  document.getElementById('view-contrib').classList.add('active');
}

function switchToContrib() {
  if (gameActive) gameStop();
  if (liveActive) liveStop();
  document.getElementById('view-game').classList.remove('active');
  document.getElementById('view-live').classList.remove('active');
  document.getElementById('view-contrib').classList.add('active');
}

/* ─── HEALTH CHECK ─── */
window.addEventListener('load', async () => {
  try {
    const d = await (await fetch(`${SERVER}/health`)).json();
    const ok = d.model_loaded;
    const cls  = ok ? 'ok'  : 'err';
    const text = ok ? 'model ready' : 'no model';
    setBadge('splash-badge',  cls, text);
    setBadge('live-badge',    cls, text);
    setBadge('game-badge',    cls, text);
    setBadge('contrib-badge', cls, text);
    if (ok) {
      setTimeout(() => {
        setBadge('splash-badge',  '', 'ready');
        setBadge('live-badge',    '', 'idle');
        setBadge('game-badge',    '', 'idle');
        setBadge('contrib-badge', '', 'idle');
      }, 2500);
    }
  } catch(e) {
    setBadge('splash-badge',  'err', 'no server');
    setBadge('live-badge',    'err', 'no server');
    setBadge('game-badge',    'err', 'no server');
    setBadge('contrib-badge', 'err', 'no server');
  }
});
