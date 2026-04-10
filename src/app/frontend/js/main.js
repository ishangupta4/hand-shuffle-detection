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

/* ─── BACKGROUND SKELETON CANVAS ─── */

const _HAND_PTS = [
  [0, 0],
  [-.18, -.12], [-.30, -.22], [-.38, -.36], [-.42, -.50],
  [-.08, -.38], [-.08, -.58], [-.08, -.72], [-.08, -.82],
  [.02, -.40],  [.02, -.62],  [.02, -.78],  [.02, -.90],
  [.12, -.36],  [.14, -.56],  [.14, -.70],  [.14, -.82],
  [.22, -.28],  [.26, -.44],  [.28, -.56],  [.28, -.65],
];

const _HAND_CONN = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

let _skelRaf = null;

function _drawSkeletonHand(ctx, cx, cy, scale, angle, r, g, b, alpha) {
  const cos = Math.cos(angle), sin = Math.sin(angle);
  const pts = _HAND_PTS.map(([x, y]) => [
    cx + (x * cos - y * sin) * scale,
    cy + (x * sin + y * cos) * scale,
  ]);

  ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
  ctx.lineWidth = 1.5;
  ctx.lineCap = 'round';
  _HAND_CONN.forEach(([a, b]) => {
    ctx.beginPath();
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
    ctx.stroke();
  });

  ctx.fillStyle = `rgba(${r},${g},${b},${alpha * 0.55})`;
  pts.forEach(([px, py], i) => {
    ctx.beginPath();
    ctx.arc(px, py, i === 0 ? 2.8 : 1.6, 0, Math.PI * 2);
    ctx.fill();
  });
}

function startBgSkeleton() {
  const canvas = document.getElementById('bg-skeleton');
  if (!canvas) return;
  if (_skelRaf) cancelAnimationFrame(_skelRaf);

  function frame(t) {
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    if (!w || !h) { _skelRaf = requestAnimationFrame(frame); return; }

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    const splash = document.getElementById('view-splash');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    if (!splash || !splash.classList.contains('active')) {
      _skelRaf = requestAnimationFrame(frame);
      return;
    }

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const [r, g, b] = isDark ? [234, 227, 216] : [26, 21, 18];
    const baseAlpha = isDark ? 0.16 : 0.09;
    const T = t * 0.00055;
    const scale = Math.min(w, h) * 0.28;

    // Left hand
    const lAlpha = baseAlpha * (0.65 + Math.sin(T * 0.8) * 0.35);
    _drawSkeletonHand(
      ctx,
      w * 0.14, h * 0.54,
      scale * (0.92 + Math.sin(T * 1.1) * 0.08),
      -0.32 + Math.sin(T * 0.9) * 0.10,
      r, g, b, lAlpha
    );

    // Right hand (mirrored — flip x coords by negating x component)
    const rAlpha = baseAlpha * (0.65 + Math.sin(T * 0.7 + 1.4) * 0.35);
    ctx.save();
    ctx.translate(w * 0.86 * 2, 0);
    ctx.scale(-1, 1);
    _drawSkeletonHand(
      ctx,
      w * 0.86, h * 0.50,
      scale * (0.92 + Math.sin(T * 1.3 + 0.9) * 0.08),
      -0.32 + Math.sin(T * 0.9 + 1.2) * 0.10,
      r, g, b, rAlpha
    );
    ctx.restore();

    _skelRaf = requestAnimationFrame(frame);
  }

  _skelRaf = requestAnimationFrame(frame);
}

/* ─── HEALTH CHECK ─── */
window.addEventListener('load', async () => {
  initTheme();
  startBgSkeleton();

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
