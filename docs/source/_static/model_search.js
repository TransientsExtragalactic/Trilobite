/*
 * Search and filter for the model directory (reference/models/directory.rst).
 *
 * Loads _static/model_index.json, which docs/source/_ext/model_index.py writes at build time from the Model
 * subclasses themselves. The widget lives in catalog_search.js. While a query or filter is active the browse
 * sections (.trilobite-model-browse) are hidden and replaced by a flat result list. Parameter, variable and
 * output names are searchable, e.g. "epsilon_B" or "flux_density".
 */
trilobiteCatalogSearch({
  mountId: "model-search",
  indexFile: "model_index.json",
  noun: "models",
  placeholder: "Search {n} models… try “broken power law”, “synchrotron”, “epsilon_B”",
  hideSelector: ".trilobite-model-browse",
  filters: [
    { label: "Category", values: function (r) { return [r.category]; } },
    { label: "Status", values: function (r) { return [r.status_label]; } },
  ],
  haystack: function (r) {
    return [r.name, r.description, r.category, r.group, r.status, r.params.join(" "), r.outputs.join(" "), r.ref].join(" ");
  },
  renderRow: function (r, row, root, el) {
    var a = el("a", "gallery-row-title", r.name);
    a.href = root + r.url;
    a.title = r.description;
    row.appendChild(a);
    var tags = el("span", "gallery-row-tags");
    tags.appendChild(el("span", "model-status model-status-" + r.status, r.status_label));
    row.appendChild(tags);
    row.appendChild(el("span", "gallery-row-meta", r.category + " · " + r.group));
  },
});
