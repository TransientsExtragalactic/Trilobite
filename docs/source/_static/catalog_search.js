/*
 * Shared search-and-filter widget for catalog landing pages (example gallery, model directory).
 *
 * `trilobiteCatalogSearch(cfg)` mounts into <div id=cfg.mountId> if present (a no-op on every other page), loads
 * _static/<cfg.indexFile> and renders a search box, one chip row per entry of cfg.filters, and a flat result
 * list. While a query or filter is active, every element matching cfg.hideSelector is hidden.
 *
 * cfg = {
 *   mountId, indexFile, noun,            // e.g. "gallery-search", "gallery_index.json", "examples"
 *   placeholder,                         // search-box placeholder; "{n}" is replaced by the record count
 *   hideSelector,                        // browse-mode elements to hide while searching
 *   filters: [{label, values: row -> [string]}],
 *   haystack: row -> string,             // text searched by the query
 *   renderRow: (row, rowEl, root, el) -> void   // fills one result row (el(tag, cls, text) is a node helper)
 * }
 */
(function () {
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  window.trilobiteCatalogSearch = function (cfg) {
    document.addEventListener("DOMContentLoaded", function () {
      var mount = document.getElementById(cfg.mountId);
      if (!mount) return;

      // Sphinx >= 7 exposes the path back to the docs root as data-content_root on <html>; older versions used
      // DOCUMENTATION_OPTIONS.URL_ROOT. Without this the fetch breaks on every page below the site root.
      var root = document.documentElement.dataset.content_root ||
        (window.DOCUMENTATION_OPTIONS && DOCUMENTATION_OPTIONS.URL_ROOT) || "";

      fetch(root + "_static/" + cfg.indexFile)
        .then(function (r) { return r.json(); })
        .then(init)
        .catch(function () { mount.style.display = "none"; });

      function init(rows) {
        var selected = cfg.filters.map(function () { return {}; });

        var input = el("input", "gallery-search-input");
        input.type = "search";
        input.placeholder = cfg.placeholder.replace("{n}", rows.length);
        input.setAttribute("aria-label", "Search " + cfg.noun);
        mount.appendChild(input);

        cfg.filters.forEach(function (f, i) {
          var items = [];
          rows.forEach(function (r) {
            f.values(r).forEach(function (v) { if (items.indexOf(v) < 0) items.push(v); });
          });
          var row = el("div", "gallery-chips");
          row.appendChild(el("span", "gallery-chip-label", f.label));
          items.forEach(function (item) {
            var b = el("button", "gallery-chip", item);
            b.type = "button";
            b.addEventListener("click", function () {
              selected[i][item] = !selected[i][item];
              b.classList.toggle("on", !!selected[i][item]);
              render();
            });
            row.appendChild(b);
          });
          mount.appendChild(row);
        });

        var count = el("div", "gallery-result-count");
        var list = el("div", "gallery-results");
        mount.appendChild(count);
        mount.appendChild(list);

        var hidden = document.querySelectorAll(cfg.hideSelector);

        function anyOn(sel) { return Object.keys(sel).some(function (k) { return sel[k]; }); }

        function render() {
          var terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
          var active = terms.length > 0 || selected.some(anyOn);

          hidden.forEach(function (s) { s.style.display = active ? "none" : ""; });
          list.textContent = "";
          if (!active) { count.textContent = ""; return; }

          var hits = rows.filter(function (r) {
            var ok = cfg.filters.every(function (f, i) {
              return !anyOn(selected[i]) || f.values(r).some(function (v) { return selected[i][v]; });
            });
            if (!ok) return false;
            var hay = cfg.haystack(r).toLowerCase();
            return terms.every(function (t) { return hay.indexOf(t) >= 0; });
          });

          count.textContent = hits.length + " of " + rows.length + " " + cfg.noun;
          if (!hits.length) { list.appendChild(el("p", "gallery-empty", "No " + cfg.noun + " match.")); return; }

          hits.forEach(function (r) {
            var rowEl = el("div", "gallery-row");
            cfg.renderRow(r, rowEl, root, el);
            list.appendChild(rowEl);
          });
        }

        input.addEventListener("input", render);
      }
    });
  };
})();
