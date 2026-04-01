/* ─── STATE ─── */

// live mode
let mpL = null, streamL = null, rafL = null;
let liveActive = false, liveTimer = null;
let liveHandsIn = false, liveHandCount = 0, liveQuality = 1.0;
let _liveStartId = 0;  // incremented on liveStop() to cancel in-flight async starts

// game mode
let mpG = null, streamG = null, rafG = null;
let gameActive = false, gameTimer = null, gameInterval = null;
let gameElapsed = 0, gameFinished = false;
let gameHandsIn = false, gameHandCount = 0, gameQuality = 1.0;
let _gameStartId = 0;  // incremented on gameStop() to cancel in-flight async starts