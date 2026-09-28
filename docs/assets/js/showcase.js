'use strict';

(() => {
  const rail = document.querySelector('.showcase-viewport');
  const track = document.querySelector('.showcase-track');
  const toggle = document.getElementById('showcase-toggle');
  const dialog = document.getElementById('media-viewer');
  const content = document.getElementById('viewer-content');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const pagination = document.querySelector('.showcase-pagination');
  let enabled = !reduced.matches;
  let visible = false;
  let hovered = false;
  let focused = false;
  let last = 0;
  let elapsed = 0;
  let index = 0;
  let origin;
  const originals = [...track.children];
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
  });
  function select(next) {
    index = (next + originals.length) % originals.length;
    elapsed = 0;
    originals.forEach((figure, i) => {
      figure.classList.toggle('is-active', i === index);
      figure.setAttribute('aria-hidden', String(i !== index));
      figure.inert = i !== index;
      const button = pagination.children[i];
      button.setAttribute('aria-pressed', String(i === index));
      button.style.setProperty('--progress', '0');
    });
    track.dataset.active = String(index);
    document.getElementById('showcase-count').textContent = `${String(index + 1).padStart(2, '0')} / ${String(originals.length).padStart(2, '0')}`;
  }
  function controls() {
    toggle.setAttribute('aria-pressed', String(enabled));
    toggle.setAttribute('aria-label', enabled ? 'Pause showcase' : 'Play showcase');
    toggle.title = toggle.getAttribute('aria-label');
    toggle.querySelector('span').className = `rx-icon icon-${enabled ? 'pause' : 'play'}`;
  }
  function frame(now) {
    const delta = last ? Math.min(now-last, 100) : 0;
    last = now;
    if (enabled && visible && !hovered && !focused && !document.hidden && !dialog.open) {
      elapsed += delta;
      if (elapsed >= 4800) select(index + 1);
      pagination.children[index].style.setProperty('--progress', String(elapsed / 4800));
    }
    requestAnimationFrame(frame);
  }
  toggle.addEventListener('click', () => { enabled = !enabled; controls(); });
  rail.addEventListener('pointerenter', () => { hovered = true; });
  rail.addEventListener('pointerleave', () => { hovered = false; });
  rail.addEventListener('focusin', () => { focused = true; });
  rail.addEventListener('focusout', event => { focused = rail.contains(event.relatedTarget); });
  rail.addEventListener('keydown', event => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault(); select(index + (event.key === 'ArrowRight' ? 1 : -1));
    }
  });
  let touchX;
  rail.addEventListener('touchstart', event => { touchX = event.changedTouches[0].clientX; }, {passive: true});
  rail.addEventListener('touchend', event => {
    const delta = event.changedTouches[0].clientX - touchX;
    if (Math.abs(delta) > 50) select(index + (delta < 0 ? 1 : -1));
  }, {passive: true});
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; }, {threshold: .3}).observe(rail);
  document.addEventListener('visibilitychange', () => { last = 0; });
  reduced.addEventListener('change', () => { if (reduced.matches) { enabled = false; controls(); } });
  document.querySelectorAll('[data-viewer]').forEach(link => {
    link.setAttribute('aria-haspopup', 'dialog');
    link.setAttribute('aria-controls', 'media-viewer');
    link.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey) return;
    event.preventDefault();
    origin = link;
    document.dispatchEvent(new Event('mimicx-media-open'));
    document.querySelectorAll('#research video').forEach(video => video.pause());
    content.replaceChildren();
    const media = document.createElement(link.dataset.viewer === 'video' ? 'video' : 'img');
    media.src = link.href;
    const caption = link.closest('figure')?.querySelector('figcaption strong')?.textContent || link.closest('figure')?.querySelector('figcaption')?.textContent || link.closest('li')?.querySelector('span')?.textContent || 'MimicX research media';
    if (media.tagName === 'VIDEO') { media.controls = true; media.playsInline = true; media.muted = true; }
    else media.alt = link.querySelector('img')?.alt || caption;
    document.getElementById('viewer-caption').textContent = caption;
    content.append(media);
    media.addEventListener('error', () => {
      const message = document.createElement('p');
      message.textContent = 'This media could not be loaded. Close the viewer and try again.';
      content.replaceChildren(message);
    }, {once:true});
    dialog.showModal();
    document.getElementById('viewer-close').focus({preventScroll:true});
    if (media.tagName === 'VIDEO') media.play().catch(() => {});
    });
  });
  document.getElementById('viewer-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    const r = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom)) dialog.close();
  });
  dialog.addEventListener('close', () => { content.querySelector('video')?.pause(); content.replaceChildren(); origin?.focus({preventScroll: true}); });
  controls();
  select(0);
  requestAnimationFrame(frame);
})();
