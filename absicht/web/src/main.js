import './style.css';
import { SECTIONS_HTML } from './sections.js';
import {
  drawThreeToOne, drawInclusion, drawSampleComplexity, drawFederation,
  drawCascade, drawMinLoop, drawQuiescence, drawRouteAudit, drawVerifFloor,
  populateTable,
} from './charts.js';

const SECTION_IDS = [
  'hero', 'problem', 'acquisition', 'composition', 'verification',
  'new-result', 'architecture', 'validation', 'discussion',
];

function mount() {
  const app = document.getElementById('app');
  app.innerHTML = `<a class="skiplink" href="#main">Skip to content</a>` + SECTIONS_HTML;
}

function initNav() {
  const sections = SECTION_IDS.map(id => document.getElementById(id)).filter(Boolean);
  const links = Array.from(document.querySelectorAll('#toc a'));
  links.forEach((a, i) => {
    const lbl = document.createElement('span');
    lbl.className = 'toc-label';
    lbl.textContent = a.dataset.label;
    a.appendChild(lbl);
    a.addEventListener('click', e => {
      e.preventDefault();
      sections[i].scrollIntoView({ behavior: 'smooth' });
    });
  });

  function onScroll() {
    const doc = document.documentElement;
    const pct = doc.scrollTop / (doc.scrollHeight - doc.clientHeight) * 100;
    const fill = document.getElementById('rail-fill');
    if (fill) fill.style.height = Math.min(100, Math.max(0, pct)) + '%';

    let activeIdx = 0;
    sections.forEach((s, i) => { if (s.getBoundingClientRect().top < window.innerHeight * 0.4) activeIdx = i; });
    links.forEach((a, i) => a.classList.toggle('active', i === activeIdx));
  }
  document.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
}

function initCounters() {
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = +el.dataset.count;
    let n = 0;
    const step = Math.max(1, Math.round(target / 40));
    const t = setInterval(() => {
      n = Math.min(target, n + step);
      el.textContent = n;
      if (n >= target) clearInterval(t);
    }, 18);
  });
}

async function loadData() {
  const res = await fetch('/data/validation.json');
  if (!res.ok) throw new Error(`Failed to load validation data: ${res.status}`);
  return res.json();
}

function drawAllCharts(DATA) {
  drawThreeToOne();
  drawInclusion(DATA);
  drawSampleComplexity(DATA);
  drawFederation(DATA);
  drawCascade(DATA);
  drawMinLoop(DATA);
  drawQuiescence(DATA);
  drawRouteAudit(DATA);
  drawVerifFloor(DATA);
}

async function boot() {
  mount();
  initNav();
  initCounters();
  populateTable();

  try {
    const DATA = await loadData();
    drawAllCharts(DATA);
    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => drawAllCharts(DATA), 200);
    });
  } catch (err) {
    console.error(err);
  }
}

boot();
