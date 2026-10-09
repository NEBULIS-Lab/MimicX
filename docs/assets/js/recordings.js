'use strict';

(() => {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const dialog = document.getElementById('media-viewer');
  const rows = [...document.querySelectorAll('.recording-row')].map(element => ({
    element, videos: [...element.querySelectorAll('video')],
    button: element.querySelector('.recording-toggle'), visible: false,
    userPaused: reduced.matches, request: 0, updating: false,
    programmaticSeeks: new WeakMap()
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

  function syncRow(row, source = row.videos[0]) {
    if (source.readyState < 1) return;
    row.videos.forEach(video => {
      if (video === source || video.readyState < 1) return;
      if (Math.abs(video.currentTime - source.currentTime) <= .08) return;
      const target = Math.min(source.currentTime, video.duration);
      row.programmaticSeeks.set(video, target);
      video.currentTime = target;
    });
  }

  async function update(row) {
    const request = ++row.request;
    row.updating = true;
    if (!eligible(row)) {
      row.videos.forEach(video => video.pause());
      row.updating = false;
      label(row);
      return;
    }
    syncRow(row);
    await Promise.allSettled(row.videos.map(video => video.play()));
    if (request !== row.request) return;
    if (!eligible(row)) row.videos.forEach(video => video.pause());
    syncRow(row);
    row.updating = false;
    label(row);
  }

  rows.forEach(row => {
    row.button.addEventListener('click', () => {
      row.userPaused = row.videos.some(video => !video.paused);
      update(row);
    });
    row.videos.forEach(video => {
      video.addEventListener('play', () => {
        if (!row.updating && row.userPaused && !reduced.matches) {
          row.userPaused = false;
          syncRow(row, video);
          update(row);
        }
        label(row);
      });
      video.addEventListener('pause', () => {
        if (!row.updating && eligible(row) && !video.ended && !video.seeking) {
          row.userPaused = true;
          update(row);
        }
        label(row);
      });
      video.addEventListener('seeking', () => {
        const expected = row.programmaticSeeks.get(video);
        row.programmaticSeeks.delete(video);
        if (expected !== undefined && Math.abs(video.currentTime - expected) < .001) return;
        syncRow(row, video);
      });
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
  // Source video is the shared clock, including each loop boundary.
  setInterval(() => rows.forEach(row => {
    if (eligible(row) && !row.updating && row.videos.every(video => !video.paused && !video.seeking)) syncRow(row);
  }), 200);
  document.addEventListener('visibilitychange', () => rows.forEach(update));
  document.addEventListener('mimicx-media-open', () => {
    rows.forEach(row => { row.request += 1; row.videos.forEach(video => video.pause()); label(row); });
  });
  dialog.addEventListener('close', () => rows.forEach(update));
  reduced.addEventListener('change', () => {
    rows.forEach(row => { row.userPaused = reduced.matches; update(row); });
  });
})();
