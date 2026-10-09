'use strict';

(() => {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const dialog = document.getElementById('media-viewer');
  const rows = [...document.querySelectorAll('.recording-row')].map(element => ({
    element, videos: [...element.querySelectorAll('video')],
    button: element.querySelector('.recording-toggle'), visible: false,
    userPaused: reduced.matches, request: 0
  }));

  function label(row) {
    const playing = row.videos.some(video => !video.paused);
    const title = `${playing ? 'Pause' : 'Play'} recordings`;
    row.button.title = title;
    row.button.setAttribute('aria-label', title);
    row.button.setAttribute('aria-pressed', String(playing));
    row.button.querySelector('.rx-icon').className = `rx-icon icon-${playing ? 'pause' : 'play'}`;
  }

  function eligible(row) {
    return row.visible && !row.userPaused && !document.hidden && !dialog.open;
  }

  async function update(row) {
    const request = ++row.request;
    if (!eligible(row)) {
      row.videos.forEach(video => video.pause());
      label(row);
      return;
    }
    // Different source lengths retain their native timing and loop independently.
    await Promise.allSettled(row.videos.map(video => video.play()));
    if (request !== row.request) return;
    if (!eligible(row)) row.videos.forEach(video => video.pause());
    label(row);
  }

  rows.forEach(row => {
    row.button.addEventListener('click', () => {
      row.userPaused = row.videos.some(video => !video.paused);
      update(row);
    });
    row.videos.forEach(video => {
      video.addEventListener('play', () => label(row));
      video.addEventListener('pause', () => label(row));
    });
    label(row);
  });
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const row = rows.find(item => item.element === entry.target);
      row.visible = entry.isIntersecting && entry.intersectionRatio >= .12;
      update(row);
    });
  }, {threshold: [0, .12]});
  rows.forEach(row => observer.observe(row.element));
  document.addEventListener('visibilitychange', () => rows.forEach(update));
  document.addEventListener('mimicx-media-open', () => {
    rows.forEach(row => { row.request += 1; row.videos.forEach(video => video.pause()); label(row); });
  });
  dialog.addEventListener('close', () => rows.forEach(update));
  reduced.addEventListener('change', () => {
    rows.forEach(row => { row.userPaused = reduced.matches; update(row); });
  });
})();
