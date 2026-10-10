'use strict';

(() => {
  const rail = document.querySelector('.showcase-viewport');
  const track = document.querySelector('.showcase-track');
  const toggle = document.getElementById('showcase-toggle');
  const dialog = document.getElementById('media-viewer');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const pagination = document.querySelector('.showcase-pagination');
  const AUTOPLAY_MS = 4000;
  const FIRST_VIDEO_MS = 8000;
  rail.dataset.interval = String(AUTOPLAY_MS);
  let enabled = !reduced.matches;
  let visible = false;
  let focused = false;
  let dragging = false;
  let moving = false;
  let paintFrame = 0;
  let autoplayFrame = 0;
  let last = 0;
  let elapsed = 0;
  let videoTime = null;
  const playbackBlocked = new WeakSet();
  let index = 0;
  const originals = [...track.children];
  const photoGaps = originals.map(() => 0);
  let workflowRequest = 0;
  let shownWorkflow = 'tennis';
  const workflowStages = document.getElementById('workflow-stages');
  const workflowButtons = [...document.querySelectorAll('[data-workflow]')];
  const workflow = {
    tennis: [
      ['showcase/tennis-stage-input.webp', 'Input video', 'Human demonstration'],
      ['showcase/tennis-stage-human.webp', 'Human reconstruction', 'GVHMR / SMPL-X'],
      ['showcase/tennis-stage-simulation.webp', 'Policy in simulation', 'G1 motion tracking'],
      ['showcase/tennis-stage-scene.webp', 'Reconstructed scene', 'Registered policy motion']
    ],
    forest: [
      ['cases/forest-stage-input.webp', 'Input video', 'Forest Traversal'],
      ['cases/forest-stage-human.webp', 'Human reconstruction', 'Recovered articulation'],
      ['cases/forest-stage-reference.webp', 'Robot reference', 'Retargeted G1 motion'],
      ['cases/forest-stage-policy.webp', 'Policy execution', 'Collision-scene rollout']
    ]
  };
  workflowButtons.forEach(button => button.addEventListener('click', async () => {
    const current = ++workflowRequest;
    const key = button.dataset.workflow;
    workflowButtons.forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    workflowStages.setAttribute('aria-busy', 'true');
    // Commit all four decoded frames together; a newer selection always wins.
    try {
      await Promise.all(workflow[key].map(async ([file]) => {
        const image = new Image();
        image.src = `assets/media/${file}`;
        await image.decode();
      }));
    } catch (_) {
      if (current !== workflowRequest) return;
      workflowStages.setAttribute('aria-busy', 'false');
      workflowButtons.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.workflow === shownWorkflow)));
      document.getElementById('workflow-note').textContent = 'These frames could not be loaded. Please try again.';
      return;
    }
    if (current !== workflowRequest) return;
    workflowStages.querySelectorAll('li').forEach((li, i) => {
      const [file, label, detail] = workflow[key][i];
      li.querySelector('a').href = `assets/media/${file}`;
      li.querySelector('img').src = `assets/media/${file}`;
      li.querySelector('img').alt = `${button.textContent}: ${label}, paper-selected frame`;
      li.querySelector('span').textContent = label;
      li.querySelector('small').textContent = detail;
      const image = li.querySelector('a');
      image.getAnimations().forEach(animation => animation.cancel());
      if (!reduced.matches && shownWorkflow !== key) image.animate(
        [{opacity:.4, transform:'translateY(6px)'}, {opacity:1, transform:'translateY(0)'}],
        {duration:260, delay:i * 35, easing:'cubic-bezier(.2,.7,.2,1)', fill:'backwards'});
    });
    workflowStages.setAttribute('aria-busy', 'false');
    shownWorkflow = key;
    document.getElementById('workflow-note').textContent = key === 'tennis'
      ? 'Tennis Swing: the four corresponding stages selected for the paper.'
      : 'Forest Traversal: paper-selected midpoint frames from each stage, not a shared physical timestamp.';
  }));
  originals.forEach((figure, i) => {
    const button = document.createElement('button');
    button.type = 'button';
    const caption = figure.querySelector('figcaption');
    button.textContent = caption.firstChild.textContent.trim();
    button.setAttribute('aria-controls', 'showcase-slides');
    button.addEventListener('click', () => select(i));
    pagination.append(button);
    figure.setAttribute('role', 'group');
    figure.setAttribute('aria-roledescription', 'slide');
    figure.setAttribute('aria-label', `${i + 1} of ${originals.length}: ${button.textContent}`);
    figure.querySelector('a').addEventListener('click', event => {
      if (i !== index) {
        event.preventDefault(); event.stopImmediatePropagation(); select(i);
      }
    });
  });
  const carousel = EmblaCarousel(rail, {loop:true, align:'center', duration:26});
  function paint() {
    const railBox = rail.getBoundingClientRect();
    const center = railBox.left + railBox.width / 2;
    const boxes = originals.map(figure => figure.getBoundingClientRect());
    const minScale = innerWidth <= 700 ? .94 : .88;
    boxes.forEach((box, i) => {
      const distance = (box.left + box.width / 2 - center) / box.width;
      const focus = Math.max(0, 1 - Math.abs(distance));
      const weight = focus * focus * (3 - 2 * focus);
      const side = Math.max(-1, Math.min(1, distance));
      const style = originals[i].style;
      style.setProperty('--album-scale', String(minScale + (1 - minScale) * weight));
      style.setProperty('--album-opacity', String(.55 + .45 * weight));
      style.setProperty('--album-tilt', `${reduced.matches ? 0 : -side * 3}deg`);
      style.setProperty('--album-shift', `${-side * photoGaps[i]}px`);
      style.setProperty('--caption-offset', `${reduced.matches ? 0 : (1 - weight) * 6}px`);
      style.setProperty('--caption-opacity', String(.3 + .7 * weight));
    });
  }
  function requestPaint() {
    if (paintFrame) return;
    paintFrame = requestAnimationFrame(() => { paintFrame = 0; paint(); });
  }
  function fitPhotos() {
    originals.forEach((figure, i) => {
      const link = figure.querySelector('a'), media = figure.querySelector('img, video');
      photoGaps[i] = Math.max(0, (link.clientWidth - media.clientWidth) / 2);
    });
    requestPaint();
  }
  originals.forEach(figure => {
    const media = figure.querySelector('img, video');
    media.addEventListener(media.tagName === 'VIDEO' ? 'loadedmetadata' : 'load', fitPhotos);
  });
  function updateSelection() {
    const previous = index;
    index = carousel.selectedScrollSnap();
    elapsed = 0;
    videoTime = null;
    const video = originals[index].querySelector('video');
    rail.dataset.interval = String(index === 0 && video ? FIRST_VIDEO_MS : AUTOPLAY_MS);
    if (video && previous !== index) {
      try { video.currentTime = 0; } catch (_) { /* Metadata may still be loading. */ }
    }
    originals.forEach((figure, i) => {
      figure.classList.toggle('is-active', i === index);
      figure.classList.toggle('is-previous', i === (index + originals.length - 1) % originals.length);
      figure.classList.toggle('is-next', i === (index + 1) % originals.length);
      figure.setAttribute('aria-hidden', String(i !== index));
      figure.querySelector('a').tabIndex = i === index ? 0 : -1;
      const button = pagination.children[i];
      button.setAttribute('aria-pressed', String(i === index));
      button.style.setProperty('--progress', '0');
    });
    track.dataset.active = String(index);
    document.getElementById('showcase-count').textContent = `${String(index + 1).padStart(2, '0')} / ${String(originals.length).padStart(2, '0')}`;
    fitPhotos();
    syncVideos();
    const button = pagination.children[index];
    const box = button.getBoundingClientRect(), container = pagination.getBoundingClientRect();
    if (box.left < container.left || box.right > container.right) pagination.scrollTo({
      left:pagination.scrollLeft + box.left - container.left - (container.width - box.width) / 2,
      behavior:reduced.matches ? 'instant' : 'smooth'
    });
  }
  function select(next) {
    const target = (next + originals.length) % originals.length;
    if (target !== index && !reduced.matches) setMoving(true);
    carousel.scrollTo(target, reduced.matches);
    elapsed = 0;
  }
  function setMoving(value) {
    moving = value;
    rail.dataset.moving = String(value);
    syncAutoplay();
  }
  carousel.on('select', updateSelection).on('reInit', updateSelection).on('scroll', requestPaint).on('settle', requestPaint);
  carousel.on('scroll', () => { if (!reduced.matches && !moving) setMoving(true); });
  carousel.on('settle', () => { last = 0; setMoving(false); });
  carousel.on('reInit', () => setMoving(false));
  carousel.on('pointerDown', () => { dragging = true; elapsed = 0; syncAutoplay(); });
  carousel.on('pointerUp', () => { dragging = false; elapsed = 0; syncAutoplay(); });
  document.getElementById('showcase-prev').addEventListener('click', () => select(index - 1));
  document.getElementById('showcase-next').addEventListener('click', () => select(index + 1));
  function controls() {
    toggle.setAttribute('aria-pressed', String(enabled));
    toggle.setAttribute('aria-label', enabled ? 'Pause showcase' : 'Play showcase');
    toggle.title = toggle.getAttribute('aria-label');
    toggle.querySelector('span').className = `rx-icon icon-${enabled ? 'pause' : 'play'}`;
  }
  function canAdvance() {
    return enabled && visible && !focused && !dragging && !moving && !document.hidden && !dialog.open;
  }
  function syncVideos() {
    originals.forEach((figure, i) => {
      const video = figure.querySelector('video');
      if (!video) return;
      const shouldPlay = () => i === index && enabled && visible && !reduced.matches && !document.hidden && !dialog.open;
      if (shouldPlay()) {
        if (video.paused && !playbackBlocked.has(video)) video.play().then(() => {
          if (!shouldPlay()) video.pause();
        }).catch(error => {
          if (error.name !== 'AbortError') playbackBlocked.add(video);
        });
      } else video.pause();
    });
  }
  function syncAutoplay() {
    syncVideos();
    if (canAdvance()) {
      if (!autoplayFrame) autoplayFrame = requestAnimationFrame(frame);
    } else {
      cancelAnimationFrame(autoplayFrame);
      autoplayFrame = 0;
      last = 0;
      videoTime = null;
    }
    rail.dataset.playing = String(canAdvance());
  }
  function frame(now) {
    autoplayFrame = 0;
    if (!canAdvance()) { syncAutoplay(); return; }
    const delta = last ? Math.min(now-last, 100) : 0;
    last = now;
    const video = originals[index].querySelector('video');
    const interval = index === 0 && video ? FIRST_VIDEO_MS : AUTOPLAY_MS;
    if (video && !video.error && !playbackBlocked.has(video)) {
      // Loading and buffering do not consume the clip's visible playback time.
      if (videoTime !== null && !video.paused && !video.seeking && video.readyState >= 3) {
        let played = video.currentTime - videoTime;
        if (played < 0 && video.loop && Number.isFinite(video.duration)) played += video.duration;
        elapsed += Math.max(0, Math.min(played * 1000, 250));
      }
      videoTime = video.currentTime;
    } else elapsed += delta;
    if (elapsed >= interval) select(index + 1);
    pagination.children[index].style.setProperty('--progress', String(elapsed / interval));
    syncAutoplay();
  }
  toggle.addEventListener('click', () => { enabled = !enabled; controls(); syncAutoplay(); });
  rail.addEventListener('focusin', event => { focused = event.target.matches(':focus-visible'); syncAutoplay(); });
  rail.addEventListener('focusout', event => {
    focused = rail.contains(event.relatedTarget) && event.relatedTarget.matches(':focus-visible');
    syncAutoplay();
  });
  rail.addEventListener('keydown', event => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault(); select(index + (event.key === 'ArrowRight' ? 1 : -1));
    }
  });
  new IntersectionObserver(entries => { visible = entries[0].intersectionRatio >= .3; syncAutoplay(); }, {threshold: .3}).observe(rail);
  document.addEventListener('visibilitychange', syncAutoplay);
  new MutationObserver(syncAutoplay).observe(dialog, {attributes:true, attributeFilter:['open']});
  reduced.addEventListener('change', () => {
    if (reduced.matches) {
      enabled = false; controls();
      carousel.scrollTo(index, true);
      setMoving(false);
      workflowStages.getAnimations({subtree:true}).forEach(animation => animation.cancel());
    }
    syncAutoplay(); requestPaint();
  });
  controls();
  updateSelection();
  setMoving(false);
  syncAutoplay();
})();
