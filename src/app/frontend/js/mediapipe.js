/* ─── MEDIAPIPE SETUP ───
   Drive MP directly via requestAnimationFrame instead of the Camera utility.
   The Camera utility caps its own internal rate; rAF runs every display frame
   (~60fps) so keypoints update at full screen refresh rate.
*/
function makeMP(onResults) {
  const h = new Hands({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
  h.setOptions({ maxNumHands:2, modelComplexity:1, minDetectionConfidence:.55, minTrackingConfidence:.5 });
  h.onResults(onResults);
  return h;
}

function startRaf(videoEl, mpInstance) {
  let running = true;
  async function loop() {
    if (!running) return;
    if (videoEl.readyState >= 2) {        // HAVE_CURRENT_DATA or better
      await mpInstance.send({ image: videoEl });
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
  return () => { running = false; };     // returns stop function
}

/* ─── DRAW HANDS ───
   lm.visibility doesn't exist in the JS Hands API (it's a Pose landmark
   property). globalAlpha manipulation is avoided to keep canvas state clean.
*/
const TIPS = new Set([4, 8, 12, 16, 20]);

function drawHands(canvas, res) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const detected = !!(res.multiHandLandmarks && res.multiHandLandmarks.length > 0);
  if (!detected) return { detected: false, count: 0, quality: 0 };

  const count = res.multiHandLandmarks.length;

  res.multiHandLandmarks.forEach((lms, i) => {
    const isLeft = res.multiHandedness[i].label === 'Right'; // MP camera flip
    const cc = isLeft ? '#2563eb' : '#dc2626';
    const dc = isLeft ? '#1d4ed8' : '#b91c1c';

    drawConnectors(ctx, lms, HAND_CONNECTIONS, { color: cc + '80', lineWidth: 1.5 });

    lms.forEach((lm, j) => {
      const x = lm.x * canvas.width;
      const y = lm.y * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, TIPS.has(j) ? 5.5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = TIPS.has(j) ? cc : dc;
      ctx.fill();
      // Canvas has CSS scaleX(-1); apply inverse transform so text renders readable
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.font = '8px DM Mono, monospace';
      ctx.fillStyle = 'rgba(255,255,255,.4)';
      ctx.fillText(j, canvas.width - x - 5, y - 2);
      ctx.restore();
    });

    const wx = lms[0].x * canvas.width;
    const wy = lms[0].y * canvas.height + 22;
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.font = '500 11px DM Mono, monospace';
    ctx.fillStyle = cc;
    ctx.fillText(isLeft ? 'LEFT' : 'RIGHT', canvas.width - wx + 20, wy);
    ctx.restore();
  });

  // Quality: fraction of landmarks fully within frame (x,y in [0,1])
  let inBounds = 0, total = 0;
  res.multiHandLandmarks.forEach(lms => {
    lms.forEach(lm => {
      total++;
      if (lm.x >= 0 && lm.x <= 1 && lm.y >= 0 && lm.y <= 1) inBounds++;
    });
  });
  const quality = total > 0 ? inBounds / total : 0;

  return { detected: true, count, quality };
}