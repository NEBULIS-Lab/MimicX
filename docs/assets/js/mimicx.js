'use strict';

(() => {
  const root = document.documentElement;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const themeButton = document.getElementById('theme-toggle');
  function setTheme(theme) {
    root.dataset.theme = theme;
    const dark = theme === 'dark';
    themeButton.setAttribute('aria-pressed', String(dark));
    themeButton.setAttribute('aria-label', `Switch to ${dark ? 'light' : 'dark'} theme`);
    themeButton.title = themeButton.getAttribute('aria-label');
    themeButton.querySelector('.rx-icon').className = `rx-icon icon-${dark ? 'sun' : 'moon'}`;
    document.getElementById('theme-label').textContent = dark ? 'Dark' : 'Light';
    document.querySelectorAll('[data-plot]').forEach(img => {
      const url = `assets/media/evidence/${img.dataset.plot}-${theme}.svg`;
      img.src = url;
      img.closest('a').href = url;
    });
    document.querySelectorAll('[data-plot-mobile]').forEach(source => {
      source.srcset = `assets/media/evidence/${source.dataset.plotMobile}-${theme}.svg`;
    });
  }
  setTheme(root.dataset.theme === 'light' ? 'light' : 'dark');
  themeButton.addEventListener('click', () => {
    const theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    setTheme(theme);
    try { localStorage.setItem('mimicx-theme', theme); } catch (_) {}
  });

  const links = Array.from(document.querySelectorAll('.research-nav nav a'));
  const nav = document.querySelector('.research-nav nav');
  const sections = links.map(link => document.querySelector(link.hash));
  let navigationFrame = 0;
  let activeSection;
  function updateNavigation() {
    navigationFrame = 0;
    const threshold = document.querySelector('.research-nav').offsetHeight + 48;
    // Include the nested rollout section; navigation order differs from DOM order.
    const positions = sections.map(section => ({section, top:section.getBoundingClientRect().top}));
    const passed = positions.filter(item => item.top <= threshold).sort((a,b) => b.top-a.top);
    const current = passed[0]?.section;
    if (current === activeSection) return;
    activeSection = current;
    links.forEach(link => {
      if (current && link.hash === '#' + current.id) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
    const link = links.find(item => item.getAttribute('aria-current'));
    if (!link) return;
    const box = link.getBoundingClientRect(), container = nav.getBoundingClientRect();
    if (box.left < container.left || box.right > container.right) nav.scrollTo({
      left:nav.scrollLeft + box.left - container.left - (container.width - box.width) / 2,
      behavior:reduced.matches ? 'instant' : 'smooth'
    });
  }
  function requestNavigation() {
    if (!navigationFrame) navigationFrame = requestAnimationFrame(updateNavigation);
  }
  window.addEventListener('scroll', requestNavigation, {passive:true});
  window.addEventListener('resize', requestNavigation);
  requestNavigation();
})();
