'use strict';

(() => {
  const connection = navigator.connection;
  const toggle = document.getElementById('low-data-mode');
  const positions = new WeakMap();
  const restores = new WeakMap();
  let preference;
  try { preference = localStorage.getItem('mimicx-low-data'); } catch (_) {}
  const networkIsSlow = () => Boolean(connection?.saveData ||
    ['slow-2g', '2g', '3g'].includes(connection?.effectiveType) ||
    (connection?.downlink > 0 && connection.downlink < 2));
  let lowData = preference ? preference === 'on' : networkIsSlow();

  function poster(video) {
    if (video.dataset.poster && !video.hasAttribute('poster')) video.poster = video.dataset.poster;
  }
  const posters = new IntersectionObserver(entries => entries.forEach(entry => {
    if (entry.isIntersecting) { poster(entry.target); posters.unobserve(entry.target); }
  }), {rootMargin:'240px 0px'});
  document.querySelectorAll('video[data-poster]').forEach(video => posters.observe(video));

  function prepare(video, preload = 'auto') {
    if (!video.dataset.src) return;
    poster(video);
    video.preload = preload;
    if (video.hasAttribute('src')) return;
    const position = positions.get(video) || 0;
    const restore = () => {
      restores.delete(video);
      if (position > 0 && Number.isFinite(video.duration)) {
        video.currentTime = Math.min(position, Math.max(0, video.duration - .04));
      }
    };
    restores.set(video, restore);
    video.addEventListener('loadedmetadata', restore, {once:true});
    video.src = video.dataset.src;
    video.load();
  }

  function cancelRestore(video) {
    const restore = restores.get(video);
    if (restore) video.removeEventListener('loadedmetadata', restore);
    restores.delete(video);
  }

  function reset(video) {
    cancelRestore(video);
    positions.set(video, 0);
    if (video.readyState >= 1) video.currentTime = 0;
  }

  function release(video) {
    if (!video.hasAttribute('src')) return;
    if (video.readyState >= 1) positions.set(video, video.currentTime);
    cancelRestore(video);
    video.pause();
    video.removeAttribute('src');
    video.preload = 'none';
    video.load();
  }

  function buffered(video, seconds = .8) {
    if (video.readyState < 3 || video.seeking) return false;
    const target = Math.min(video.duration - .02, video.currentTime + seconds);
    for (let i = 0; i < video.buffered.length; i++) {
      if (video.buffered.start(i) <= video.currentTime + .03 && video.buffered.end(i) >= target) return true;
    }
    return false;
  }

  function publish() {
    toggle.checked = lowData;
    document.documentElement.dataset.lowData = String(lowData);
    document.dispatchEvent(new Event('mimicx-data-mode'));
  }
  toggle.addEventListener('change', () => {
    lowData = toggle.checked;
    preference = lowData ? 'on' : 'off';
    try { localStorage.setItem('mimicx-low-data', preference); } catch (_) {}
    publish();
  });
  connection?.addEventListener('change', () => {
    if (!preference) { lowData = networkIsSlow(); publish(); }
  });
  window.MimicXMedia = {prepare, release, reset, buffered, get lowData() { return lowData; }};
  publish();
})();
