'use strict';

(() => {
  const root = document.documentElement;
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
  }
  setTheme(root.dataset.theme === 'light' ? 'light' : 'dark');
  themeButton.addEventListener('click', () => {
    const theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    setTheme(theme);
    try { localStorage.setItem('mimicx-theme', theme); } catch (_) {}
  });

  const taskNames = {tennis: 'Tennis Swing', football: 'Football Juggling', dance: 'Dance Sequence', kungfu: 'Kung Fu'};
  const taskRows = {tennis: 'Tennis Swing', football: 'Football Juggling', dance: 'Dance Sequence', kungfu: 'Kung Fu Sequence'};
  const notes = {
    tennis: 'Tennis reaches strict success in all nine MimicX evaluation rollouts, compared with one of nine for Fixed Reference.',
    football: 'Execution horizon increases from 53.7 to 427.3 steps. Neither method completes the full verification budget in this cohort.',
    dance: 'Repeated verification retains the incumbent policy. The selected output improves body tracking and execution horizon over Fixed Reference.',
    kungfu: 'The retained policy reduces mean body error from 0.249 m to 0.141 m. Verification preserves this policy when continuation candidates do not pass.'
  };
  const fixed = document.getElementById('fixed-video');
  const ours = document.getElementById('ours-video');
  const videos = [fixed, ours];
  const playButton = document.getElementById('pair-toggle');
  const seek = document.getElementById('pair-seek');
  const status = document.getElementById('playback-status');
  const output = document.getElementById('pair-time');
  const tabs = Array.from(document.querySelectorAll('[data-task]'));
  let generation = 0;
  let playingTogether = false;
  let starting = false;

  function duration() { return Math.min(...videos.map(v => v.duration)); }
  function updatePlayButton(playing) {
    playButton.querySelector('.rx-icon').className = `rx-icon icon-${playing ? 'pause' : 'play'}`;
    playButton.setAttribute('aria-label', playing ? 'Pause both videos' : 'Play both videos');
    playButton.title = playButton.getAttribute('aria-label');
  }
  function updateTime() {
    const length = duration();
    output.value = `${fixed.currentTime.toFixed(1)} s`;
    if (Number.isFinite(length) && length > 0) seek.value = String(Math.min(fixed.currentTime / length, 1) * 1000);
  }
  function stopPair() {
    playingTogether = false;
    videos.forEach(video => video.pause());
    updatePlayButton(false);
  }
  function selectTask(task, focus = false) {
    generation += 1;
    stopPair();
    starting = false;
    playButton.disabled = false;
    status.textContent = '';
    tabs.forEach(tab => {
      const selected = tab.dataset.task === task;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focus) tab.focus();
    });
    document.getElementById('policy-panel').setAttribute('aria-labelledby', `task-${task}`);
    videos.forEach((video, index) => {
      const method = index === 0 ? 'fixed' : 'ours';
      video.poster = `assets/media/research/${task}-${method}.jpg`;
      video.src = `assets/media/${task}-${method}.mp4`;
      video.setAttribute('aria-label', `${index === 0 ? 'Fixed Reference' : 'MimicX'} ${taskNames[task]} rollout`);
      video.load();
    });
    seek.value = '0';
    output.value = '0.0 s';
    document.getElementById('task-note').textContent = notes[task];
    const rows = Array.from(document.querySelectorAll('#core-results tr')).filter(row => row.cells[0].textContent === taskRows[task]);
    if (rows.length === 2) {
      document.getElementById('task-error').textContent = rows.map(row => row.cells[4].textContent.replace(' m', '')).join(' / ') + ' m';
      document.getElementById('task-horizon').textContent = rows.map(row => row.cells[3].textContent).join(' / ') + ' steps';
    }
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => selectTask(tab.dataset.task));
    tab.addEventListener('keydown', event => {
      let target;
      if (event.key === 'ArrowRight') target = (i + 1) % tabs.length;
      if (event.key === 'ArrowLeft') target = (i + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') target = 0;
      if (event.key === 'End') target = tabs.length - 1;
      if (target !== undefined) { event.preventDefault(); selectTask(tabs[target].dataset.task, true); }
    });
  });
  function ready(video) {
    if (video.readyState >= 1) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => finish(new Error('Timed out')), 15000);
      function finish(error) {
        clearTimeout(timer);
        video.removeEventListener('loadedmetadata', success);
        video.removeEventListener('error', failure);
        video.removeEventListener('emptied', cancelled);
        if (error) reject(error); else resolve();
      }
      const success = () => finish();
      const failure = () => finish(new Error('Video unavailable'));
      const cancelled = () => finish(new Error('Task changed'));
      video.addEventListener('loadedmetadata', success);
      video.addEventListener('error', failure);
      video.addEventListener('emptied', cancelled);
      video.preload = 'auto';
    });
  }
  playButton.addEventListener('click', async () => {
    if (videos.some(v => !v.paused)) { stopPair(); return; }
    const current = generation;
    starting = true;
    playButton.disabled = true;
    status.textContent = 'Loading recorded rollouts...';
    try {
      await Promise.all(videos.map(ready));
      if (generation !== current) return;
      const length = duration();
      if (fixed.currentTime >= length - 0.05) fixed.currentTime = 0;
      ours.currentTime = fixed.currentTime;
      await Promise.all(videos.map(v => v.play()));
      if (generation !== current) return;
      playingTogether = true;
      updatePlayButton(true);
      status.textContent = '';
    } catch (_) {
      if (generation === current) {
        stopPair();
        status.textContent = 'Playback could not start. Retry or use the individual video controls.';
      }
    } finally {
      if (generation === current) { playButton.disabled = false; starting = false; }
    }
  });
  document.getElementById('pair-reset').addEventListener('click', () => {
    generation += 1;
    starting = false;
    playButton.disabled = false;
    stopPair();
    videos.forEach(v => { if (v.readyState >= 1) v.currentTime = 0; });
    status.textContent = '';
    seek.value = '0';
    output.value = '0.0 s';
  });
  seek.addEventListener('input', () => {
    const length = duration();
    if (Number.isFinite(length)) {
      videos.forEach(v => { v.currentTime = Number(seek.value) / 1000 * length; });
      output.value = `${fixed.currentTime.toFixed(1)} s`;
    }
  });
  fixed.addEventListener('timeupdate', () => {
    updateTime();
    if (playingTogether && !fixed.paused && !ours.paused && Math.abs(fixed.currentTime - ours.currentTime) > 0.2) ours.currentTime = fixed.currentTime;
  });
  videos.forEach(video => {
    video.addEventListener('ended', stopPair);
    video.addEventListener('pause', () => { if (playingTogether && !starting) stopPair(); });
    video.addEventListener('error', () => {
      stopPair();
      status.textContent = 'This recording could not be loaded. Please retry.';
    });
  });
  const links = Array.from(document.querySelectorAll('.research-nav nav a'));
  const observer = new IntersectionObserver(entries => {
    const current = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (!current) return;
    links.forEach(link => {
      if (link.hash === '#' + current.target.id) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }, {rootMargin: '-15% 0px -60% 0px'});
  document.querySelectorAll('#research > section').forEach(section => observer.observe(section));
})();
