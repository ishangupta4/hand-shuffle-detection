/* ─── NAVIGATION ─── */
function switchToGame() {
  document.getElementById('view-live').classList.remove('active');
  document.getElementById('view-game').classList.add('active');
}

function switchToLive() {
  if (gameActive) gameStop();
  document.getElementById('view-game').classList.remove('active');
  document.getElementById('view-live').classList.add('active');
}

/* ─── HEALTH CHECK ─── */
window.addEventListener('load', async () => {
  try {
    const d = await (await fetch(`${SERVER}/health`)).json();
    const ok = d.model_loaded;
    setBadge('live-badge', ok ? 'ok' : 'err', ok ? 'model ready' : 'no model');
    setBadge('game-badge', ok ? 'ok' : 'err', ok ? 'model ready' : 'no model');
    if (ok) setTimeout(() => { setBadge('live-badge', '', 'idle'); setBadge('game-badge', '', 'idle'); }, 2500);
  } catch(e) {
    setBadge('live-badge', 'err', 'no server');
    setBadge('game-badge', 'err', 'no server');
  }
});