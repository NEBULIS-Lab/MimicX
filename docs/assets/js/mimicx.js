'use strict';
const fixed = document.getElementById('fixed-video');
const ours = document.getElementById('ours-video');
const pairToggle = document.getElementById('pair-toggle');
const seek = document.getElementById('pair-seek');
const notes = {
  tennis: 'Tennis: MimicX reaches strict success in all nine core evaluation rollouts. Videos illustrate a selected recorded trial.',
  football: 'Football juggling: the mean worst first-failure horizon increases from 53.7 to 427.3 steps. Strict full-budget success remains 0% in this cohort.',
  dance: 'Dance: verification protects the registered current policy. The comparison shows the selected output, not an accepted candidate for this task.',
  kungfu: 'Kung Fu: verification protects the registered current policy. The sequence remains challenging; the full task and component results are retained in the data.'
};
function duration() { return Math.min(fixed.duration, ours.duration); }
function stopPair() { fixed.pause(); ours.pause(); pairToggle.textContent = 'Play both'; }
document.getElementById('task-select').addEventListener('change', event => {
  stopPair(); const task = event.target.value;
  fixed.src = `assets/media/${task}-fixed.mp4`; ours.src = `assets/media/${task}-ours.mp4`;
  fixed.load(); ours.load(); seek.value = '0'; document.getElementById('pair-time').value = '0.0 s';
  document.getElementById('task-note').textContent = notes[task];
});
pairToggle.addEventListener('click', async () => {
  if (!fixed.paused || !ours.paused) { stopPair(); return; }
  if (fixed.readyState < 1 || ours.readyState < 1) {
    await Promise.all([fixed, ours].map(video => video.readyState >= 1 ? Promise.resolve() : new Promise(resolve => video.addEventListener('loadedmetadata', resolve, {once:true}))));
  }
  ours.currentTime = fixed.currentTime;
  try { await Promise.all([fixed.play(), ours.play()]); pairToggle.textContent = 'Pause both'; }
  catch { stopPair(); }
});
document.getElementById('pair-reset').addEventListener('click', () => { stopPair(); fixed.currentTime = 0; ours.currentTime = 0; seek.value = '0'; });
seek.addEventListener('input', () => { const length = duration(); if (Number.isFinite(length)) fixed.currentTime = ours.currentTime = Number(seek.value) / 1000 * length; });
fixed.addEventListener('timeupdate', () => {
  const length = duration(); if (Number.isFinite(length) && length > 0) seek.value = String(fixed.currentTime / length * 1000);
  document.getElementById('pair-time').value = `${fixed.currentTime.toFixed(1)} s`;
  if (!fixed.paused && !ours.paused && Math.abs(fixed.currentTime - ours.currentTime) > 0.25) ours.currentTime = fixed.currentTime;
});
for (const video of [fixed, ours]) video.addEventListener('ended', stopPair);
