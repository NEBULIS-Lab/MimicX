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
  let mediaSize;
  let closeTimer;
  let viewerMotion;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');

  function carryViewer(closing = false) {
    const interrupted = viewerMotion?.playState === 'running'
      ? {transform:getComputedStyle(dialog).transform, opacity:getComputedStyle(dialog).opacity} : null;
    viewerMotion?.cancel();
    if (reduced.matches) return;
    const box = (origin.querySelector('img, video') || origin).getBoundingClientRect();
    const destination = dialog.getBoundingClientRect();
    const visible = box.bottom > 0 && box.top < innerHeight && box.width > 0;
    const dx = visible ? box.left + box.width / 2 - destination.left - destination.width / 2 : 0;
    const dy = visible ? box.top + box.height / 2 - destination.top - destination.height / 2 : 8;
    const scale = visible ? Math.max(.25, Math.min(.98, box.width / destination.width, box.height / destination.height)) : .985;
    const source = {transform:`translate(${dx}px,${dy}px) scale(${scale})`, opacity:0};
    const target = {transform:'translate(0px,0px) scale(1)', opacity:1};
    // The same thumbnail anchors both directions; interruption starts at the current pose.
    viewerMotion = dialog.animate(closing ? [interrupted || target, source] : [source, target], {
      duration:closing ? 220 : 320, easing:closing ? 'cubic-bezier(.4,0,.8,.4)' : 'cubic-bezier(.16,1,.3,1)', fill:'both'
    });
    viewerMotion.id = 'media-carry';
  }
  function openViewer() {
    document.documentElement.classList.add('media-open');
    dialog.showModal();
    const thumbnail = origin.querySelector('img');
    if (thumbnail?.naturalWidth) {
      mediaSize = {width:thumbnail.naturalWidth, height:thumbnail.naturalHeight};
      sizeViewer();
    }
    carryViewer();
  }
  function closeViewer() {
    if (!dialog.open || dialog.classList.contains('viewer-closing')) return;
    content.querySelector('video')?.pause();
    if (reduced.matches) { dialog.close(); return; }
    dialog.classList.add('viewer-closing');
    carryViewer(true);
    closeTimer = setTimeout(() => dialog.close(), 220);
  }

  function sizeViewer() {
    if (!mediaSize || !dialog.open) return;
    const padding = getComputedStyle(content);
    const gutterX = parseFloat(padding.paddingLeft) + parseFloat(padding.paddingRight);
    const gutterY = parseFloat(padding.paddingTop) + parseFloat(padding.paddingBottom);
    const viewportWidth = document.documentElement.clientWidth;
    const maxWidth = Math.min(viewportWidth <= 700 ? viewportWidth - 12 : viewportWidth * .94, 1280);
    const maxHeight = Math.min(innerHeight - 12, innerHeight * .94);
    let width = maxWidth;
    let mediaWidth;
    let mediaHeight;
    // The toolbar can wrap on a narrow portrait image, so include its measured height.
    for (let pass = 0; pass < 4; pass += 1) {
      dialog.style.setProperty('--viewer-width', `${width}px`);
      dialog.classList.toggle('viewer-compact', width < 620 || innerWidth <= 700);
      const toolbarHeight = dialog.querySelector('.viewer-toolbar').offsetHeight;
      const scale = Math.min((maxWidth - gutterX - 2) / mediaSize.width,
        Math.max(80, maxHeight - toolbarHeight - gutterY - 2) / mediaSize.height);
      mediaWidth = mediaSize.width * scale;
      mediaHeight = mediaSize.height * scale;
      width = Math.min(maxWidth, Math.max(280, mediaWidth + gutterX + 2));
    }
    dialog.style.setProperty('--viewer-width', `${width}px`);
    dialog.style.setProperty('--viewer-height', `${mediaHeight}px`);
    const stage = content.querySelector('.viewer-stage') || content.querySelector('video');
    stage.style.width = `${mediaWidth}px`;
    stage.style.marginInline = 'auto';
  }

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
    const img = target.querySelector('img');
    mediaSize = img ? {width:img.naturalWidth, height:img.naturalHeight}
      : {width:target.offsetWidth, height:target.offsetHeight};
    sizeViewer();
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
    clearTimeout(closeTimer);
    viewerMotion?.cancel(); viewerMotion = null;
    dialog.classList.remove('viewer-closing');
    request += 1;
    panzoom?.destroy(); panzoom = null; surface = null; mediaSize = null;
    dialog.style.removeProperty('--viewer-width');
    dialog.style.removeProperty('--viewer-height');
    dialog.classList.remove('viewer-compact');
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
      const caption = link.dataset.caption || (type === 'table' ? 'Four-task comparison'
        : link.closest('figure')?.querySelector('figcaption strong')?.textContent
          || link.closest('figure')?.querySelector('figcaption')?.textContent
          || link.closest('li')?.querySelector('span')?.textContent
          || link.getAttribute('title') || 'MimicX research media');
      document.getElementById('viewer-caption').textContent = caption;
      zoomTools.hidden = type === 'video';
      if (type === 'video') {
        const video = document.createElement('video');
        video.addEventListener('loadedmetadata', () => {
          if (current !== request || !dialog.open) return;
          mediaSize = {width:video.videoWidth, height:video.videoHeight};
          sizeViewer();
        }, {once:true});
        video.src = link.href; video.controls = true; video.playsInline = true; video.muted = true;
        content.append(video);
        openViewer();
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
          openViewer(); enableZoom(target);
        } else {
          const img = document.createElement('img');
          img.alt = link.querySelector('img')?.alt || caption;
          img.addEventListener('load', () => {
            if (current === request && dialog.open) enableZoom(target);
          }, {once:true});
          img.addEventListener('error', () => {
            if (current === request) { target.textContent = 'Image unavailable. Close the viewer and try again.'; zoomTools.hidden = true; }
          }, {once:true});
          target.append(img); openViewer(); img.src = link.href;
        }
      }
      document.getElementById('viewer-close').focus({preventScroll:true});
    });
  });
  document.getElementById('viewer-plus').addEventListener('click', () => panzoom?.zoomIn());
  document.getElementById('viewer-minus').addEventListener('click', () => panzoom?.zoomOut());
  document.getElementById('viewer-reset').addEventListener('click', fit);
  document.getElementById('viewer-close').addEventListener('click', closeViewer);
  dialog.addEventListener('cancel', event => { event.preventDefault(); closeViewer(); });
  dialog.addEventListener('click', event => {
    const r = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom)) closeViewer();
  });
  dialog.addEventListener('close', () => {
    document.documentElement.classList.remove('media-open');
    clear(); origin?.focus({preventScroll:true});
  });
  reduced.addEventListener('change', () => {
    if (reduced.matches) {
      viewerMotion?.cancel();
      if (dialog.classList.contains('viewer-closing')) dialog.close();
    }
  });
  window.addEventListener('resize', () => {
    if (dialog.open) requestAnimationFrame(() => {
      if (!dialog.classList.contains('viewer-closing')) viewerMotion?.cancel();
      sizeViewer(); fit();
    });
  });
})();
