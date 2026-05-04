/* ─── THEME ─── */

function initTheme() {
  const saved = localStorage.getItem('hsa-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const dark = saved ? saved === 'dark' : prefersDark;
  if (dark) document.documentElement.setAttribute('data-theme', 'dark');
}

function toggleTheme() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  if (isDark) {
    document.documentElement.removeAttribute('data-theme');
    localStorage.setItem('hsa-theme', 'light');
  } else {
    document.documentElement.setAttribute('data-theme', 'dark');
    localStorage.setItem('hsa-theme', 'dark');
  }
}

function _applyThemeIcons() {
  // CSS handles toggle appearance via [data-theme="dark"] on <html>
}

/* ─── NAVIGATION ─── */

function switchView(toId) {
  const toEl = document.getElementById(toId);
  if (!toEl) return;

  const fromEl = document.querySelector('.view.active');
  if (!fromEl || fromEl === toEl) {
    toEl.classList.add('active');
    return;
  }

  fromEl.style.transition = 'opacity .15s ease';
  fromEl.style.opacity = '0';

  setTimeout(() => {
    fromEl.classList.remove('active');
    fromEl.style.transition = '';
    fromEl.style.opacity = '';

    toEl.style.opacity = '0';
    toEl.classList.add('active');

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        toEl.style.transition = 'opacity .22s ease';
        toEl.style.opacity = '1';
        setTimeout(() => {
          toEl.style.transition = '';
          toEl.style.opacity = '';
        }, 240);
      });
    });
  }, 160);
}

function switchFromSplashToGame() {
  switchView('view-game');
}

function switchFromSplashToLive() {
  switchView('view-live');
}

function switchToSplash() {
  if (typeof gameActive !== 'undefined' && gameActive) gameStop();
  if (typeof liveActive !== 'undefined' && liveActive) liveStop();
  if (typeof contribRecording !== 'undefined' && contribRecording) contribStop();
  switchView('view-splash');
}

function switchToContribReview() {
  switchView('view-contrib-review');
}

function switchToGame() {
  if (typeof liveActive !== 'undefined' && liveActive) liveStop();
  switchView('view-game');
}

function switchToLive() {
  if (typeof gameActive !== 'undefined' && gameActive) gameStop();
  switchView('view-live');
}

function switchFromSplashToContrib() {
  switchView('view-contrib');
}

function switchToContrib() {
  if (typeof gameActive !== 'undefined' && gameActive) gameStop();
  if (typeof liveActive !== 'undefined' && liveActive) liveStop();
  switchView('view-contrib');
}

/* ─── HEALTH CHECK ─── */
window.addEventListener('load', async () => {
  initTheme();
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
