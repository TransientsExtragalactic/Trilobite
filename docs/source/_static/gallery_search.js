/*
 * Search and filter for the example-gallery landing page (examples.rst).
 *
 * Mounts into <div id="gallery-search"> if present (a no-op on every other page) and loads
 * _static/gallery_index.json, which docs/source/_ext/gallery_index.py writes at build time.
 * While a query or filter is active the stage sections (.trilobite-stage) are hidden and replaced by
 * a flat result list.
 */
document.addEventListener("DOMContentLoaded", function () {
  var mount = document.getElementById("gallery-search");
  if (!mount) return;

  var root = (window.DOCUMENTATION_OPTIONS && DOCUMENTATION_OPTIONS.URL_ROOT) || "";

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  fetch(root + "_static/gallery_index.json")
    .then(function (r) { return r.json(); })
    .then(init)
    .catch(function () { mount.style.display = "none"; });

  function init(rows) {
    var topics = [], cases = [];
    rows.forEach(function (r) {
      if (topics.indexOf(r.topic) < 0) topics.push(r.topic);
      r.cases.forEach(function (c) { if (cases.indexOf(c) < 0) cases.push(c); });
    });

    var selTopic = {}, selCase = {};

    var input = el("input", "gallery-search-input");
    input.type = "search";
    input.placeholder = "Search " + rows.length + " examples… try “cooling”, “TDE”, “censored”";
    input.setAttribute("aria-label", "Search examples");
    mount.appendChild(input);

    function chipRow(label, items, sel) {
      var row = el("div", "gallery-chips");
      row.appendChild(el("span", "gallery-chip-label", label));
      items.forEach(function (item) {
        var b = el("button", "gallery-chip", item);
        b.type = "button";
        b.addEventListener("click", function () {
          sel[item] = !sel[item];
          b.classList.toggle("on", !!sel[item]);
          render();
        });
        row.appendChild(b);
      });
      mount.appendChild(row);
      return row;
    }
    chipRow("Topic", topics, selTopic);
    chipRow("Science case", cases, selCase);

    var count = el("div", "gallery-result-count");
    var list = el("div", "gallery-results");
    mount.appendChild(count);
    mount.appendChild(list);

    var stages = document.querySelectorAll(".trilobite-stage");

    function anyOn(sel) { return Object.keys(sel).some(function (k) { return sel[k]; }); }

    function render() {
      var terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
      var active = terms.length > 0 || anyOn(selTopic) || anyOn(selCase);

      stages.forEach(function (s) { s.style.display = active ? "none" : ""; });
      list.textContent = "";
      if (!active) { count.textContent = ""; return; }

      var hits = rows.filter(function (r) {
        if (anyOn(selTopic) && !selTopic[r.topic]) return false;
        if (anyOn(selCase) && !r.cases.some(function (c) { return selCase[c]; })) return false;
        var hay = (r.title + " " + r.topic + " " + r.sub + " " + r.cases.join(" ")).toLowerCase();
        return terms.every(function (t) { return hay.indexOf(t) >= 0; });
      });

      count.textContent = hits.length + " of " + rows.length + " examples";
      if (!hits.length) { list.appendChild(el("p", "gallery-empty", "No examples match.")); return; }

      hits.forEach(function (r) {
        var row = el("div", "gallery-row");
        var a = el("a", "gallery-row-title", r.title);
        a.href = root + r.url;
        row.appendChild(a);
        var tags = el("span", "gallery-row-tags");
        r.cases.forEach(function (c) { tags.appendChild(el("span", "gallery-tag", c)); });
        row.appendChild(tags);
        row.appendChild(el("span", "gallery-row-meta", r.topic + " · " + r.sub));
        list.appendChild(row);
      });
    }

    input.addEventListener("input", render);
  }
});
