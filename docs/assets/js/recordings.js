'use strict';

(() => {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const dialog = document.getElementById('media-viewer');
  const media = window.MimicXMedia;
  let viewerOpening = false;
  const pauses = new WeakSet(), plays = new WeakSet();
  const loadButtons = new WeakMap();
  const rows = [...document.querySelectorAll('.recording-row')].map(element => ({
    element, videos:[...element.querySelectorAll('video')],
    button:element.querySelector('.recording-toggle'), visible:false, near:false,
    userPaused:reduced.matches, starting:false, buffering:false, request:0,
    programmaticSeeks:new WeakMap()
  }));

  function pause(video, unload = false) {
    if (!video.paused) pauses.add(video);
    video.pause();
    if (unload) {
      media.release(video);
      // load() cancels queued media events; do not suppress the next user action.
      pauses.delete(video);
      plays.delete(video);
    }
  }

  function label(row) {
    row.videos.forEach(video => {
      const loaded = video.hasAttribute('src');
      video.controls = loaded;
      const button = loadButtons.get(video);
      if (button) button.hidden = loaded;
    });
    const playing = row.videos.some(video => !video.paused) || row.starting || row.buffering;
    const title = `${playing ? 'Pause' : 'Play'} recordings`;
    row.button.title = title;
    row.button.setAttribute('aria-label', title);
    row.button.setAttribute('aria-pressed', String(playing));
    row.button.querySelector('.rx-icon').className = `rx-icon icon-${playing ? 'pause' : 'play'}`;
  }

  function eligible(row) {
    return row.visible && !row.userPaused && !document.hidden && !dialog.open && !viewerOpening;
  }

  function syncRow(row, source = row.videos[0]) {
    if (row.element.dataset.sync === 'independent' || source.readyState < 1) return;
    row.videos.forEach(video => {
      if (video === source || video.readyState < 1 || Math.abs(video.currentTime - source.currentTime) <= .08) return;
      const target = Math.min(source.currentTime, Math.max(0, video.duration - .04));
      row.programmaticSeeks.set(video, target);
      video.currentTime = target;
    });
  }

  function update(row) {
    const wanted = row.videos;
    const blocked = document.hidden || dialog.open || viewerOpening;
    if (!eligible(row)) {
      row.request++;
      row.starting = false;
      row.buffering = false;
      row.videos.forEach(video => pause(video, blocked || !row.visible));
      if (!blocked && row.near && !row.visible && !row.userPaused) {
        media.prepare(row.element.querySelector('[data-method="ours"] video'), 'metadata');
      }
      label(row);
      return;
    }
    wanted.forEach(video => media.prepare(video));
    if (row.starting) return;
    if (wanted.every(video => !video.paused && video.readyState >= 3)) {
      row.buffering = false;
      if (wanted.every(video => !video.seeking)) syncRow(row);
      label(row);
      return;
    }
    // Hold the shared timeline when any recording needs more data.
    wanted.forEach(video => pause(video));
    row.buffering = !wanted.every(video => media.buffered(video));
    if (row.buffering) { label(row); return; }
    syncRow(row);
    if (wanted.some(video => video.seeking)) return;
    row.starting = true;
    const request = ++row.request;
    Promise.allSettled(wanted.map(video => {
      plays.add(video);
      return video.play().catch(error => { plays.delete(video); throw error; });
    })).then(() => {
      if (request !== row.request) return;
      row.starting = false;
      if (!eligible(row)) wanted.forEach(video => pause(video));
      label(row);
    });
    label(row);
  }

  rows.forEach(row => {
    row.button.addEventListener('click', () => {
      row.userPaused = row.starting || row.buffering || row.videos.some(video => !video.paused);
      update(row);
    });
    row.videos.forEach(video => {
      const choose = () => {
        if (!video.hasAttribute('src')) {
          row.request++;
          row.starting = false;
          row.userPaused = false;
          update(row);
        }
      };
      if (video.parentElement.dataset.crop === 'descent') {
        // Follow the descent within the square source without cutting head or feet.
        const crop = () => {
          const progress = Number.isFinite(video.duration) && video.duration > 0
            ? Math.min(1, Math.max(0, video.currentTime / video.duration)) : 0;
          video.style.setProperty('--crop-y', `${25 + 20 * progress}%`);
        };
        ['timeupdate', 'loadedmetadata', 'emptied'].forEach(event => video.addEventListener(event, crop));
      }
      // An unloaded native Play control cannot emit a play event in all browsers.
      const loadButton = document.createElement('button');
      loadButton.type = 'button';
      loadButton.className = 'recording-load rx-icon-button';
      loadButton.title = `Play ${video.getAttribute('aria-label') || 'recording'}`;
      loadButton.setAttribute('aria-label', loadButton.title);
      loadButton.innerHTML = '<span class="rx-icon icon-play" aria-hidden="true"></span>';
      loadButton.addEventListener('click', choose);
      video.parentElement.append(loadButton);
      loadButtons.set(video, loadButton);
      video.tabIndex = 0;
      video.addEventListener('click', choose);
      video.addEventListener('keydown', event => {
        if (!video.hasAttribute('src') && [' ', 'Enter'].includes(event.key)) {
          event.preventDefault(); choose();
        }
      });
      video.addEventListener('play', () => {
        if (plays.delete(video)) { label(row); return; }
        row.userPaused = false;
        update(row);
      });
      video.addEventListener('pause', () => {
        if (pauses.delete(video)) { label(row); return; }
        if (eligible(row) && !video.ended && !video.seeking) {
          row.userPaused = true;
          update(row);
        }
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
  const observer = new IntersectionObserver(entries => entries.forEach(entry => {
    const row = rows.find(item => item.element === entry.target);
    row.visible = entry.isIntersecting && entry.intersectionRatio >= .12;
    update(row);
  }), {threshold:[0, .12]});
  const near = new IntersectionObserver(entries => entries.forEach(entry => {
    const row = rows.find(item => item.element === entry.target);
    row.near = entry.isIntersecting;
    update(row);
  }), {rootMargin:'120px 0px'});
  rows.forEach(row => { observer.observe(row.element); near.observe(row.element); });
  setInterval(() => rows.filter(row => row.visible).forEach(update), 250);
  document.addEventListener('visibilitychange', () => rows.forEach(update));
  document.addEventListener('mimicx-media-open', () => { viewerOpening = true; rows.forEach(update); });
  dialog.addEventListener('close', () => { viewerOpening = false; rows.forEach(update); });
  reduced.addEventListener('change', () => {
    rows.forEach(row => { row.userPaused = reduced.matches; update(row); });
  });
})();
