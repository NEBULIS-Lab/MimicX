'use strict';

(() => {
  const rail = document.querySelector('.showcase-viewport');
  const track = document.querySelector('.showcase-track');
  const toggle = document.getElementById('showcase-toggle');
  const dialog = document.getElementById('media-viewer');
  const content = document.getElementById('viewer-content');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const mobile = matchMedia('(max-width: 700px)');
  let enabled = !reduced.matches && !mobile.matches;
  let visible = false;
  let hovered = false;
  let focused = false;
  let suspendedUntil = 0;
  let last = 0;
  let lastVideoCheck = 0;
  let origin;
  let activeVideo;
  const originals = [...track.children];
  originals.forEach(figure => {
    const clone = figure.cloneNode(true);
    clone.setAttribute('aria-hidden', 'true');
    clone.dataset.clone = '';
    clone.querySelectorAll('a').forEach(a => { a.tabIndex = -1; });
    track.append(clone);
  });
  const videos = [...track.querySelectorAll('video')];
  function controls() {
    toggle.setAttribute('aria-pressed', String(enabled));
    toggle.setAttribute('aria-label', enabled ? 'Pause showcase' : 'Play showcase');
    toggle.title = toggle.getAttribute('aria-label');
    toggle.querySelector('span').className = `rx-icon icon-${enabled ? 'pause' : 'play'}`;
  }
  function pauseVideos() { videos.forEach(v => v.pause()); activeVideo = null; }
  function chooseVideo() {
    if (!enabled || !visible || document.hidden || dialog.open || reduced.matches) { pauseVideos(); return; }
    const box = rail.getBoundingClientRect();
    const ranked = videos.map(video => {
      const rect = video.getBoundingClientRect();
      return {video, overlap: Math.max(0, Math.min(rect.right, box.right) - Math.max(rect.left, box.left)) / rect.width};
    }).sort((a, b) => b.overlap - a.overlap);
    const next = ranked[0]?.overlap > .5 ? ranked[0].video : null;
    if (next === activeVideo) return;
    pauseVideos();
    activeVideo = next;
    if (next) {
      if (!next.getAttribute('src')) next.src = next.dataset.src;
      next.play().then(() => { if (activeVideo !== next || document.hidden || dialog.open || !visible) next.pause(); }).catch(() => {});
    }
  }
  function frame(now) {
    const delta = last ? Math.min((now-last)/1000, .08) : 0;
    last = now;
    if (enabled && visible && !hovered && !focused && !document.hidden && !dialog.open && now > suspendedUntil) {
      rail.scrollLeft += 25 * delta;
      const loopWidth = track.children[originals.length].offsetLeft - track.children[0].offsetLeft;
      if (loopWidth && rail.scrollLeft >= loopWidth) rail.scrollLeft -= loopWidth;
    }
    if (now-lastVideoCheck > 500) { chooseVideo(); lastVideoCheck = now; }
    requestAnimationFrame(frame);
  }
  toggle.addEventListener('click', () => { enabled = !enabled; controls(); chooseVideo(); });
  rail.addEventListener('pointerenter', () => { hovered = true; });
  rail.addEventListener('pointerleave', () => { hovered = false; });
  rail.addEventListener('focusin', () => { focused = true; });
  rail.addEventListener('focusout', event => { focused = rail.contains(event.relatedTarget); });
  ['wheel', 'touchstart', 'pointerdown', 'keydown'].forEach(name => rail.addEventListener(name, () => { suspendedUntil = performance.now()+6000; }, {passive: true}));
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; chooseVideo(); }, {threshold: .15}).observe(rail);
  document.addEventListener('visibilitychange', chooseVideo);
  reduced.addEventListener('change', () => { if (reduced.matches) { enabled = false; controls(); pauseVideos(); } });
  mobile.addEventListener('change', () => { if (mobile.matches) { enabled = false; controls(); pauseVideos(); } });
  document.querySelectorAll('[data-viewer]').forEach(link => link.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey) return;
    event.preventDefault();
    origin = link;
    pauseVideos();
    document.querySelectorAll('#research video').forEach(video => video.pause());
    content.replaceChildren();
    const media = document.createElement(link.dataset.viewer === 'video' ? 'video' : 'img');
    media.src = link.href;
    const caption = link.closest('figure')?.querySelector('figcaption')?.textContent || link.closest('li')?.querySelector('span')?.textContent || 'MimicX research media';
    if (media.tagName === 'VIDEO') { media.controls = true; media.playsInline = true; media.muted = true; }
    else media.alt = link.querySelector('img')?.alt || caption;
    document.getElementById('viewer-caption').textContent = caption;
    content.append(media);
    dialog.showModal();
    if (media.tagName === 'VIDEO') media.play().catch(() => {});
  }));
  document.getElementById('viewer-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => { content.querySelector('video')?.pause(); content.replaceChildren(); origin?.focus({preventScroll: true}); });
  controls();
  requestAnimationFrame(frame);
})();
