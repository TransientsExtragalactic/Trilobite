/*
 * Search and filter for the example-gallery landing page (examples.rst).
 *
 * Loads _static/gallery_index.json, which docs/source/_ext/gallery_index.py writes at build time. The widget itself
 * lives in catalog_search.js. While a query or filter is active the stage sections (.trilobite-stage) are hidden
 * and replaced by a flat result list.
 */
trilobiteCatalogSearch({
  mountId: "gallery-search",
  indexFile: "gallery_index.json",
  noun: "examples",
  placeholder: "Search {n} examples… try “cooling”, “TDE”, “censored”",
  hideSelector: ".trilobite-stage",
  filters: [
    { label: "Topic", values: function (r) { return [r.topic]; } },
    { label: "Science case", values: function (r) { return r.cases; } },
  ],
  haystack: function (r) { return r.title + " " + r.topic + " " + r.sub + " " + r.cases.join(" "); },
  renderRow: function (r, row, root, el) {
    var a = el("a", "gallery-row-title", r.title);
    a.href = root + r.url;
    row.appendChild(a);
    var tags = el("span", "gallery-row-tags");
    r.cases.forEach(function (c) { tags.appendChild(el("span", "gallery-tag", c)); });
    row.appendChild(tags);
    row.appendChild(el("span", "gallery-row-meta", r.topic + " · " + r.sub));
  },
});
