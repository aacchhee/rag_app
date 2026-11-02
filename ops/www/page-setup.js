/* Shared head setup for index.html and reader.html: load it in <head>, before MathJax. */

// Theme: apply the saved appearance before first paint, so dark mode does not flash
(function () {
  var key = 'rag_user_prefs';
  var prefs = {};
  try { prefs = JSON.parse(localStorage.getItem(key) || '{}'); } catch (e) {}
  var appearance = prefs.appearance || 'system';
  var dark = appearance === 'dark' || (appearance === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  if (dark) document.documentElement.setAttribute('data-theme', 'dark');
})();

// MathJax configuration (read by the MathJax script that loads after this file)
window.MathJax = {
  loader: { load: ['a11y/semantic-enrich'] },
  tex: {
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    a11y: { speech: true, braille: false, subtitles: false }
  },
  chtml: {
    mtextInheritFont: true
  }
};
