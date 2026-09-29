'use strict';

(() => {
  const rail = document.querySelector('.showcase-viewport');
  const track = document.querySelector('.showcase-track');
  const toggle = document.getElementById('showcase-toggle');
  const dialog = document.getElementById('media-viewer');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const pagination = document.querySelector('.showcase-pagination');
  const AUTOPLAY_MS = 4000;
  rail.dataset.interval = String(AUTOPLAY_MS);
  let enabled = !reduced.matches;
  let visible = false;
  let hovered = false;
  let focused = false;
  let dragging = false;
  let paintFrame = 0;
  let last = 0;
  let elapsed = 0;
  let index = 0;
  const originals = [...track.children];
  const photoGaps = originals.map(() => 0);
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
  document.querySelectorAll('[data-workflow]').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('[data-workflow]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    document.querySelectorAll('#workflow-stages li').forEach((li, i) => {
      const [file, label, detail] = workflow[button.dataset.workflow][i];
      li.querySelector('a').href = `assets/media/${file}`;
      li.querySelector('img').src = `assets/media/${file}`;
      li.querySelector('img').alt = `${button.textContent}: ${label}, paper-selected frame`;
      li.querySelector('span').textContent = label;
      li.querySelector('small').textContent = detail;
    });
    document.getElementById('workflow-note').textContent = button.dataset.workflow === 'tennis'
      ? 'Tennis Swing: the four corresponding stages selected for the paper.'
      : 'Forest Traversal: paper-selected midpoint frames from each stage, not a shared physical timestamp.';
  }));
  originals.forEach((figure, i) => {
    const button = document.createElement('button');
    button.type = 'button';
    const caption = figure.querySelector('figcaption');
    button.textContent = i === 5 ? 'Parkour reference' : caption.firstChild.textContent.trim();
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
    });
  }
  function requestPaint() {
    if (paintFrame) return;
    paintFrame = requestAnimationFrame(() => { paintFrame = 0; paint(); });
  }
  function fitPhotos() {
    originals.forEach((figure, i) => {
      const link = figure.querySelector('a'), img = figure.querySelector('img');
      photoGaps[i] = Math.max(0, (link.clientWidth - img.clientWidth) / 2);
    });
    requestPaint();
  }
  originals.forEach(figure => figure.querySelector('img').addEventListener('load', fitPhotos));
  function updateSelection() {
    index = carousel.selectedScrollSnap();
    elapsed = 0;
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
  }
  function select(next) {
    carousel.scrollTo((next + originals.length) % originals.length, reduced.matches);
    elapsed = 0;
  }
  carousel.on('select', updateSelection).on('reInit', updateSelection).on('scroll', requestPaint).on('settle', requestPaint);
  carousel.on('pointerDown', () => { dragging = true; elapsed = 0; });
  carousel.on('pointerUp', () => { dragging = false; elapsed = 0; });
  document.getElementById('showcase-prev').addEventListener('click', () => select(index - 1));
  document.getElementById('showcase-next').addEventListener('click', () => select(index + 1));
  function controls() {
    toggle.setAttribute('aria-pressed', String(enabled));
    toggle.setAttribute('aria-label', enabled ? 'Pause showcase' : 'Play showcase');
    toggle.title = toggle.getAttribute('aria-label');
    toggle.querySelector('span').className = `rx-icon icon-${enabled ? 'pause' : 'play'}`;
  }
  function frame(now) {
    const delta = last ? Math.min(now-last, 100) : 0;
    last = now;
    if (enabled && visible && !hovered && !focused && !dragging && !document.hidden && !dialog.open) {
      elapsed += delta;
      if (elapsed >= AUTOPLAY_MS) select(index + 1);
      pagination.children[index].style.setProperty('--progress', String(elapsed / AUTOPLAY_MS));
    }
    requestAnimationFrame(frame);
  }
  toggle.addEventListener('click', () => { enabled = !enabled; controls(); });
  rail.addEventListener('pointerenter', event => { if (event.pointerType !== 'touch') hovered = true; });
  rail.addEventListener('pointerleave', () => { hovered = false; });
  rail.addEventListener('focusin', event => { focused = event.target.matches(':focus-visible'); });
  rail.addEventListener('focusout', event => { focused = rail.contains(event.relatedTarget); });
  rail.addEventListener('keydown', event => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault(); select(index + (event.key === 'ArrowRight' ? 1 : -1));
    }
  });
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; }, {threshold: .3}).observe(rail);
  document.addEventListener('visibilitychange', () => { last = 0; });
  reduced.addEventListener('change', () => { if (reduced.matches) { enabled = false; controls(); } requestPaint(); });
  controls();
  updateSelection();
  requestAnimationFrame(frame);
})();
