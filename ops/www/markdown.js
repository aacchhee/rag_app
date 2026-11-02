/* Shared Markdown rendering for index.html and reader.html (needs marked.js). */

function escapeHtml(str) {
  return String(str || "").replace(/[&<>"']/g, function (c) {
    return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c];
  });
}

/**
 * Render markdown to HTML safely.
 * Protects LaTeX delimiters from being mangled by the markdown parser.
 */
function renderMarkdown(text) {
  if (!text) return "";

  // 1. Protect LaTeX blocks by replacing them with placeholders
  var placeholders = [];
  var i = 0;

  // Protect display math ($$...$$) first
  text = text.replace(/\$\$[\s\S]*?\$\$/g, function (match) {
    var id = "%%MATHBLOCK" + (i++) + "%%";
    placeholders.push({ id: id, content: match });
    return id;
  });

  // Protect inline math ($...$)
  text = text.replace(/\$[^$\n]+?\$/g, function (match) {
    var id = "%%MATHBLOCK" + (i++) + "%%";
    placeholders.push({ id: id, content: match });
    return id;
  });

  // Protect \[...\] display math
  text = text.replace(/\\\[[\s\S]*?\\\]/g, function (match) {
    var id = "%%MATHBLOCK" + (i++) + "%%";
    placeholders.push({ id: id, content: match });
    return id;
  });

  // Protect \(...\) inline math
  text = text.replace(/\\\([\s\S]*?\\\)/g, function (match) {
    var id = "%%MATHBLOCK" + (i++) + "%%";
    placeholders.push({ id: id, content: match });
    return id;
  });

  // 2. Parse markdown to HTML
  var html = marked.parse(text, { breaks: true });

  // 3. Restore LaTeX placeholders
  // Escape < to prevent the browser treating math content like \sum_{i<k} as HTML tags.
  // A function, not a string: in a replacement string "$$" means "$", which
  // would turn display math into inline math.
  placeholders.forEach(function (p) {
    var restored = p.content.replace(/</g, "&lt;");
    html = html.replace(p.id, function () { return restored; });
  });

  return html;
}
