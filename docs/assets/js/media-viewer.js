'use strict';

(() => {
  const dialog = document.getElementById('media-viewer');
  const content = document.getElementById('viewer-content');
  const zoomTools = document.querySelector('.viewer-zoom');
  const scaleLabel = document.getElementById('viewer-scale');
  const zoomButtons = [...zoomTools.querySelectorAll('button')];
  let panzoom;
  let surface;
  let origin;
  let fitScale = 1;
  let request = 0;

  function fit() {
    if (!panzoom || !surface) return;
    const stage = surface.parentElement;
    fitScale = surface.classList.contains('is-table')
      ? Math.min(stage.clientWidth / surface.offsetWidth, stage.clientHeight / surface.offsetHeight, 1) : 1;
    panzoom.setOptions({minScale:fitScale});
    panzoom.zoom(fitScale, {animate:false, force:true});
    panzoom.pan(0, 0, {animate:false, force:true});
  }
  function enableZoom(target) {
    surface = target;
    panzoom = Panzoom(surface, {maxScale:5, minScale:.1, cursor:'grab', step:.3});
    surface.addEventListener('panzoomchange', event => {
      scaleLabel.textContent = `${Math.round(event.detail.scale / fitScale * 100)}%`;
      content.dataset.scale = String(event.detail.scale);
    });
    surface.parentElement.addEventListener('wheel', event => panzoom?.zoomWithWheel(event), {passive:false});
    surface.addEventListener('dblclick', () => {
      if (panzoom.getScale() > fitScale * 1.1) fit();
      else panzoom.zoom(fitScale * 2, {animate:true});
    });
    const instance = panzoom;
    requestAnimationFrame(() => {
      if (panzoom !== instance) return;
      fit();
      requestAnimationFrame(() => {
        if (panzoom !== instance) return;
        content.dataset.scale = String(panzoom.getScale());
        content.dataset.ready = 'true';
        zoomButtons.forEach(button => { button.disabled = false; });
      });
    });
  }
  function clear() {
    request += 1;
    panzoom?.destroy(); panzoom = null; surface = null;
    content.querySelector('video')?.pause();
    content.replaceChildren();
    delete content.dataset.scale;
    delete content.dataset.ready;
    zoomButtons.forEach(button => { button.disabled = true; });
    scaleLabel.textContent = '100%';
  }
  document.querySelectorAll('[data-viewer]').forEach(link => {
    link.setAttribute('aria-haspopup', 'dialog');
    link.setAttribute('aria-controls', 'media-viewer');
    link.addEventListener('click', event => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.shiftKey) return;
      event.preventDefault(); origin = link;
      clear();
      const current = request;
      const type = link.dataset.viewer;
      document.dispatchEvent(new Event('mimicx-media-open'));
      document.querySelectorAll('#research video').forEach(video => video.pause());
      const caption = type === 'table' ? 'Four-task comparison'
        : link.closest('figure')?.querySelector('figcaption strong')?.textContent
          || link.closest('figure')?.querySelector('figcaption')?.textContent
          || link.closest('li')?.querySelector('span')?.textContent
          || link.getAttribute('title') || 'MimicX research media';
      document.getElementById('viewer-caption').textContent = caption;
      zoomTools.hidden = type === 'video';
      if (type === 'video') {
        const video = document.createElement('video');
        video.src = link.href; video.controls = true; video.playsInline = true; video.muted = true;
        content.append(video);
        dialog.showModal();
        video.play().catch(() => {});
      } else {
        const stage = document.createElement('div'); stage.className = 'viewer-stage';
        const target = document.createElement('div'); target.className = 'viewer-pan-surface';
        stage.append(target); content.append(stage);
        if (type === 'table') {
          target.classList.add('is-table');
          const table = document.querySelector('#results .rx-table').cloneNode(true);
          table.querySelectorAll('[id]').forEach(node => node.removeAttribute('id'));
          target.append(table);
          dialog.showModal(); enableZoom(target);
        } else {
          const img = document.createElement('img');
          img.alt = link.querySelector('img')?.alt || caption;
          img.addEventListener('load', () => {
            if (current === request && dialog.open) enableZoom(target);
          }, {once:true});
          img.addEventListener('error', () => {
            if (current === request) { target.textContent = 'Image unavailable. Close the viewer and try again.'; zoomTools.hidden = true; }
          }, {once:true});
          target.append(img); dialog.showModal(); img.src = link.href;
        }
      }
      document.getElementById('viewer-close').focus({preventScroll:true});
    });
  });
  document.getElementById('viewer-plus').addEventListener('click', () => panzoom?.zoomIn());
  document.getElementById('viewer-minus').addEventListener('click', () => panzoom?.zoomOut());
  document.getElementById('viewer-reset').addEventListener('click', fit);
  document.getElementById('viewer-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    const r = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom)) dialog.close();
  });
  dialog.addEventListener('close', () => { clear(); origin?.focus({preventScroll:true}); });
  window.addEventListener('resize', () => { if (dialog.open) requestAnimationFrame(fit); });
})();
