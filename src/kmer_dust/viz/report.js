/* kmer-dust report runtime.
 *
 * Design notes, because the shape of this file is not obvious:
 *
 * 1. Nothing is a plotly figure baked in Python.  A run can carry 300k bins and
 *    eight different colourings; serialising a figure per colouring, or even one
 *    figure plus a parallel copy of x/y for the linking code, doubles a payload
 *    that is already the dominant cost of the file.  Instead Python emits ONE
 *    columnar blob (base64 little-endian typed arrays) and this file builds the
 *    traces.  Switching colouring is then a single Plotly.restyle of a colour
 *    array -- no re-layout, no reflow, no second copy of the coordinates.
 *
 * 2. Cross-highlighting between the map and the genome ribbon is done by
 *    rewriting the colour arrays (selected keep their colour, the rest go to the
 *    theme's dim colour) rather than by plotly's own selection machinery, which
 *    only dims within the plot that owns the selection.
 *
 * 3. Hover is ours.  Pre-rendering 300k hover strings in Python would add
 *    ~25 MB; building them lazily from the code arrays costs nothing.
 */
(function () {
  "use strict";

  var KD = window.KD || {};
  var HAS_PLOTLY = typeof Plotly !== "undefined";
  var $ = function (id) { return document.getElementById(id); };

  // ---------------------------------------------------------------- decoding

  var CTORS = {
    f4: Float32Array, f8: Float64Array,
    i1: Int8Array, i2: Int16Array, i4: Int32Array,
    u1: Uint8Array, u2: Uint16Array, u4: Uint32Array
  };

  function decode(spec) {
    if (!spec) return null;
    var bin = atob(spec.b);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    // Every platform that runs a browser is little-endian, which is the byte
    // order Python wrote.
    return new (CTORS[spec.d] || Float32Array)(bytes.buffer, 0, spec.n);
  }

  var _cache = {};
  function arr(name) {
    if (_cache[name]) return _cache[name];
    var out = decode(KD.arrays && KD.arrays[name]);
    if (out) _cache[name] = out;
    return out;
  }

  // ---------------------------------------------------------------- theme

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  // A host that embeds this report (an artifact viewer, an iframe with a
  // restrictive sandbox) may forbid downloads outright, in which case plotly's
  // "save as png" button is a control that visibly does nothing.  A host sets
  // window.KD_EMBEDDED to have it hidden; opened as a file, the button works
  // and stays.
  function embedded() { return !!window.KD_EMBEDDED; }
  function hiddenButtons(extra) {
    var out = (extra || []).slice();
    if (embedded()) out.push("toImage");
    return out;
  }

  function isDark() { return document.documentElement.getAttribute("data-theme") !== "light"; }
  function dimColor() { return isDark() ? "rgba(231,234,238,0.055)" : "rgba(23,26,31,0.06)"; }

  // ---------------------------------------------------------------- format

  function fmtInt(v) {
    if (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) return "–";
    return Math.round(v).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
  function fmtNum(v, dp) {
    if (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) return "–";
    return Number(v).toFixed(dp === undefined ? 2 : dp);
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  // ---------------------------------------------------------------- colours

  function hexToRgb(hex) {
    var h = hex.replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  function sampleScale(stops, t) {
    if (!(t >= 0)) t = 0;
    if (t > 1) t = 1;
    var i = 1;
    while (i < stops.length - 1 && stops[i][0] < t) i++;
    var a = stops[i - 1], b = stops[i];
    var span = b[0] - a[0];
    var f = span > 0 ? (t - a[0]) / span : 0;
    var ca = hexToRgb(a[1]), cb = hexToRgb(b[1]);
    return "rgb(" + Math.round(ca[0] + (cb[0] - ca[0]) * f) + "," +
      Math.round(ca[1] + (cb[1] - ca[1]) * f) + "," +
      Math.round(ca[2] + (cb[2] - ca[2]) * f) + ")";
  }

  var colorCache = {};
  function baseColors(coloring) {
    if (colorCache[coloring.key]) return colorCache[coloring.key];
    var n = KD.n, out = new Array(n), i;
    var values = arr(coloring.array);
    if (coloring.kind === "cat") {
      var colors = KD.levels[coloring.levels].colors;
      for (i = 0; i < n; i++) out[i] = colors[values[i]] || KD.missingColor;
    } else {
      // 257-entry lookup rather than interpolating per point: at 300k points the
      // difference is a visible pause when the colouring changes.
      var lo = coloring.cmin, hi = coloring.cmax, span = hi - lo || 1;
      var stops = coloring.scale;
      var lut = new Array(257), k;
      for (k = 0; k <= 256; k++) lut[k] = sampleScale(stops, k / 256);
      for (i = 0; i < n; i++) {
        var v = values[i];
        out[i] = (v === v) ? lut[Math.max(0, Math.min(256, Math.round((v - lo) / span * 256)))]
          : KD.missingColor;
      }
    }
    colorCache[coloring.key] = out;
    return out;
  }

  // ---------------------------------------------------------------- state

  var state = {
    coloring: null,
    mask: null,        // Uint8Array or null
    count: 0,
    label: "",
    legendPick: -1,
    tablePick: null
  };

  var umapDiv = $("kd-umap");
  var ribbonDiv = $("kd-ribbon");
  var heatDiv = $("kd-heatmap");
  var ribbon = KD.ribbon || null;
  var ribbonOrder = ribbon ? decode(ribbon.order) : null;
  var ribbonX = null, ribbonY = null;

  // ---------------------------------------------------------------- plots

  function axisBase() {
    return {
      showgrid: true, gridcolor: cssVar("--grid"), gridwidth: 1,
      zeroline: false, showline: false,
      tickfont: { size: 10, color: cssVar("--muted") },
      linecolor: cssVar("--line")
    };
  }

  function plotFont() {
    return { family: cssVar("--sans") || "sans-serif", size: 11, color: cssVar("--muted") };
  }

  function buildUmap() {
    if (!HAS_PLOTLY || !KD.n || !KD.hasCoords) {
      umapDiv.innerHTML = '<div class="kd-empty">' +
        (!KD.n ? "No bins to plot yet &mdash; run the <code>matrix</code> stage first."
          : !HAS_PLOTLY ? "plotly.js is unavailable, so the map cannot be drawn."
            : "No embedding yet &mdash; run <code>decompose</code> and <code>embed</code>.") +
        "</div>";
      umapDiv.style.height = "auto";
      return;
    }
    var trace = {
      type: "scattergl",
      mode: "markers",
      x: arr("x"),
      y: arr("y"),
      marker: {
        size: KD.pointSize, color: baseColors(state.coloring).slice(),
        line: { width: 0 }, opacity: 0.85
      },
      selected: { marker: { opacity: 0.95 } },
      unselected: { marker: { opacity: 0.85 } },
      hoverinfo: "none",
      showlegend: false
    };
    var layout = {
      margin: { l: 34, r: 8, t: 8, b: 28 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: plotFont(),
      xaxis: Object.assign(axisBase(), { title: { text: KD.embedLabel + " 1", font: { size: 10 } } }),
      yaxis: Object.assign(axisBase(), {
        title: { text: KD.embedLabel + " 2", font: { size: 10 } },
        scaleanchor: "x", scaleratio: 1
      }),
      dragmode: "lasso",
      hovermode: "closest"
    };
    Plotly.newPlot(umapDiv, [trace], layout, {
      responsive: true, displaylogo: false, scrollZoom: true,
      modeBarButtonsToRemove: hiddenButtons(),
      toImageButtonOptions: { format: "png", scale: 2, filename: KD.runName + "_map" }
    });
    umapDiv.on("plotly_selected", function (ev) { onScatterSelect(ev); });
    umapDiv.on("plotly_deselect", function () { clearSelection(); });
    umapDiv.on("plotly_hover", function (ev) { showTip(ev, umapDiv); });
    umapDiv.on("plotly_unhover", hideTip);
  }

  /* Squares sized so that consecutive bins just touch: the ribbon should read as
     a painted bar, not a dotted line, at whatever width the card happens to be. */
  function ribbonMarkerSize() {
    var width = ribbonDiv.clientWidth || umapDiv.clientWidth || 900;
    var perRow = ribbon.maxBinsPerRow || 1;
    return Math.max(3, Math.min(14, (width / perRow) * 2.0));
  }

  function buildRibbon() {
    if (!ribbon || !HAS_PLOTLY) {
      ribbonDiv.innerHTML = '<div class="kd-empty">No placed reference bins to lay out &mdash; ' +
        "the ribbon needs bins with a resolved chromosome name.</div>";
      $("kd-ribbon-sub").textContent = "";
      return;
    }
    var m = ribbonOrder.length;
    ribbonX = new Float64Array(m);
    ribbonY = new Float64Array(m);
    var starts = arr("start"), chrom = arr("chrom");
    var half = KD.binSize / 2e6;
    for (var j = 0; j < m; j++) {
      var p = ribbonOrder[j];
      ribbonX[j] = starts[p] / 1e6 + half;
      ribbonY[j] = ribbon.rowOfChrom[chrom[p]];
    }
    var shapes = ribbon.rows.map(function (row, i) {
      return {
        type: "rect", xref: "x", yref: "y",
        x0: 0, x1: row.length / 1e6, y0: i - 0.34, y1: i + 0.34,
        fillcolor: cssVar("--dim"), line: { width: 0 }, layer: "below"
      };
    });
    var trace = {
      type: "scattergl", mode: "markers",
      x: ribbonX, y: ribbonY,
      marker: {
        size: ribbonMarkerSize(), symbol: "square",
        color: pickColors(ribbonOrder), line: { width: 0 }
      },
      selected: { marker: { opacity: 1 } },
      unselected: { marker: { opacity: 1 } },
      hoverinfo: "none", showlegend: false
    };
    var layout = {
      margin: { l: 62, r: 12, t: 6, b: 30 },
      height: Math.max(120, 36 + 24 * ribbon.rows.length),
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: plotFont(),
      shapes: shapes,
      xaxis: Object.assign(axisBase(), {
        title: { text: "position (Mb)", font: { size: 10 } },
        showgrid: true, rangemode: "tozero"
      }),
      yaxis: {
        tickmode: "array",
        tickvals: ribbon.rows.map(function (_, i) { return i; }),
        ticktext: ribbon.rows.map(function (r) { return r.name; }),
        tickfont: { size: 10, color: cssVar("--ink-2") },
        showgrid: false, zeroline: false,
        range: [ribbon.rows.length - 0.5, -0.5],
        fixedrange: true
      },
      dragmode: "select",
      hovermode: "closest"
    };
    Plotly.newPlot(ribbonDiv, [trace], layout, {
      responsive: true, displaylogo: false,
      modeBarButtonsToRemove: hiddenButtons(["lasso2d"]),
      toImageButtonOptions: { format: "png", scale: 2, filename: KD.runName + "_ribbon" }
    });
    ribbonDiv.on("plotly_selected", function (ev) { onRibbonSelect(ev); });
    ribbonDiv.on("plotly_deselect", function () { clearSelection(); });
    ribbonDiv.on("plotly_click", function (ev) { onRibbonClick(ev); });
    ribbonDiv.on("plotly_hover", function (ev) { showTip(ev, ribbonDiv, true); });
    ribbonDiv.on("plotly_unhover", hideTip);
    $("kd-ribbon-sub").textContent = fmtInt(m) + " bins · " + ribbon.rows.length +
      (ribbon.rows.length === 1 ? " chromosome" : " chromosomes") + " · " + ribbon.subject;
  }

  function buildHeatmap() {
    var hm = KD.heatmap;
    if (!hm || !HAS_PLOTLY) {
      heatDiv.innerHTML = '<div class="kd-empty">No enrichment table &mdash; run the ' +
        "<code>enrich</code> stage to compare clusters against the annotation tracks.</div>";
      heatDiv.style.height = "auto";
      $("kd-heat-sub").textContent = "";
      return;
    }
    var trace = {
      type: "heatmap",
      z: hm.z, x: hm.x, y: hm.y, text: hm.text,
      hovertemplate: "%{text}<extra></extra>",
      zmin: -hm.zabs, zmax: hm.zabs, zmid: 0,
      colorscale: isDark() ? hm.scales.dark : hm.scales.light,
      xgap: 1, ygap: 1,
      colorbar: {
        title: { text: "log2", side: "right", font: { size: 10 } },
        thickness: 10, len: 0.55, outlinewidth: 0,
        tickfont: { size: 9, color: cssVar("--muted") }
      }
    };
    heatDiv.style.height = Math.max(240, 60 + 17 * hm.y.length) + "px";
    Plotly.newPlot(heatDiv, [trace], {
      margin: { l: 168, r: 20, t: 8, b: 116 },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: plotFont(),
      xaxis: { tickangle: -55, tickfont: { size: 10, color: cssVar("--muted") }, showgrid: false },
      // Rows arrive largest-cluster-first; plotly draws index 0 at the bottom,
      // so reverse the axis to keep the biggest cluster where the eye starts.
      yaxis: {
        tickfont: { size: 10, color: cssVar("--ink-2") }, showgrid: false,
        autorange: "reversed"
      }
    }, {
      responsive: true, displaylogo: false, displayModeBar: false
    });
    $("kd-heat-sub").textContent = hm.y.length + " clusters × " + hm.x.length + " features";
  }

  // ---------------------------------------------------------------- colours -> plots

  function pickColors(indices) {
    var base = currentColors();
    var out = new Array(indices.length);
    for (var i = 0; i < indices.length; i++) out[i] = base[indices[i]];
    return out;
  }

  var _current = null;
  function currentColors() {
    if (_current) return _current;
    var base = baseColors(state.coloring);
    if (!state.mask) { _current = base; return _current; }
    var dim = dimColor(), n = KD.n, out = new Array(n);
    for (var i = 0; i < n; i++) out[i] = state.mask[i] ? base[i] : dim;
    _current = out;
    return _current;
  }

  function plotted(gd) { return !!(gd && gd.data && gd.data.length); }

  function repaint() {
    _current = null;
    if (!plotted(umapDiv)) return;
    var colors = currentColors();
    Plotly.restyle(umapDiv, { "marker.color": [colors.slice()] }, [0]);
    if (plotted(ribbonDiv)) {
      Plotly.restyle(ribbonDiv, { "marker.color": [pickColors(ribbonOrder)] }, [0]);
    }
  }

  // ---------------------------------------------------------------- selection

  function setSelection(mask, count, label) {
    state.mask = count > 0 ? mask : null;
    state.count = count > 0 ? count : 0;
    state.label = count > 0 ? label : "";
    var status = $("kd-selstatus");
    if (state.mask) {
      status.textContent = fmtInt(state.count) + " of " + fmtInt(KD.n) + " bins" +
        (label ? " · " + label : "");
      status.classList.add("is-active");
      $("kd-clear").disabled = false;
    } else {
      status.textContent = "no selection";
      status.classList.remove("is-active");
      $("kd-clear").disabled = true;
    }
    repaint();
    renderLegend();
    markTableRows();
  }

  function clearSelection() {
    state.legendPick = -1;
    state.tablePick = null;
    setSelection(null, 0, "");
  }

  function maskFromPoints(points, map) {
    var mask = new Uint8Array(KD.n), count = 0;
    for (var i = 0; i < points.length; i++) {
      var idx = points[i].pointIndex;
      if (idx === undefined || idx === null) idx = points[i].pointNumber;
      if (idx === undefined || idx === null) continue;
      var p = map ? map[idx] : idx;
      if (p >= 0 && p < KD.n && !mask[p]) { mask[p] = 1; count++; }
    }
    return { mask: mask, count: count };
  }

  function onScatterSelect(ev) {
    if (!ev || !ev.points || !ev.points.length) { clearSelection(); return; }
    state.legendPick = -1; state.tablePick = null;
    var r = maskFromPoints(ev.points, null);
    setSelection(r.mask, r.count, "lasso");
  }

  function onRibbonSelect(ev) {
    if (!ev || !ev.points || !ev.points.length) { clearSelection(); return; }
    state.legendPick = -1; state.tablePick = null;
    var r = maskFromPoints(ev.points, ribbonOrder);
    setSelection(r.mask, r.count, describeInterval(ev.points));
  }

  function describeInterval(points) {
    var lo = Infinity, hi = -Infinity, chrom = arr("chrom"), starts = arr("start"), rows = {};
    for (var i = 0; i < points.length; i++) {
      var j = points[i].pointIndex;
      if (j === undefined) j = points[i].pointNumber;
      if (j === undefined) continue;
      var p = ribbonOrder[j];
      var st = starts[p];
      if (st < lo) lo = st;
      if (st > hi) hi = st;
      rows[KD.levels.chrom.labels[chrom[p]]] = 1;
    }
    var names = Object.keys(rows);
    if (!names.length) return "";
    var where = names.length === 1 ? names[0] : names.length + " chromosomes";
    if (names.length !== 1) return where;
    return where + ":" + (lo / 1e6).toFixed(2) + "–" +
      ((hi + KD.binSize) / 1e6).toFixed(2) + " Mb";
  }

  /* Clicking the ribbon selects a *region*, not a bin: walk outward from the
     clicked bin while the neighbours keep the same colour category (for a
     categorical colouring) or stay within a small window (for a continuous one).
     Clicking the middle of an alpha-satellite array therefore lights up that
     whole array in the map, which is the question people actually ask. */
  function onRibbonClick(ev) {
    if (!ev || !ev.points || !ev.points.length) return;
    var j = ev.points[0].pointIndex;
    if (j === undefined) j = ev.points[0].pointNumber;
    if (j === undefined) return;
    state.legendPick = -1; state.tablePick = null;
    var m = ribbonOrder.length;
    var lo = j, hi = j, MAXRUN = 20000;
    if (state.coloring.kind === "cat") {
      var codes = arr(state.coloring.array);
      var want = codes[ribbonOrder[j]];
      while (lo > 0 && ribbonY[lo - 1] === ribbonY[j] &&
        codes[ribbonOrder[lo - 1]] === want && j - lo < MAXRUN) lo--;
      while (hi < m - 1 && ribbonY[hi + 1] === ribbonY[j] &&
        codes[ribbonOrder[hi + 1]] === want && hi - j < MAXRUN) hi++;
    } else {
      lo = Math.max(0, j - 25); hi = Math.min(m - 1, j + 25);
      while (lo < j && ribbonY[lo] !== ribbonY[j]) lo++;
      while (hi > j && ribbonY[hi] !== ribbonY[j]) hi--;
    }
    var mask = new Uint8Array(KD.n), count = 0, pts = [];
    for (var q = lo; q <= hi; q++) {
      var p = ribbonOrder[q];
      if (!mask[p]) { mask[p] = 1; count++; }
      pts.push({ pointIndex: q });
    }
    setSelection(mask, count, describeInterval(pts));
  }

  function selectByLevel(levelIndex) {
    var coloring = state.coloring;
    if (coloring.kind !== "cat") return;
    if (state.legendPick === levelIndex) { clearSelection(); return; }
    var codes = arr(coloring.array), n = KD.n;
    var mask = new Uint8Array(n), count = 0;
    for (var i = 0; i < n; i++) if (codes[i] === levelIndex) { mask[i] = 1; count++; }
    state.legendPick = levelIndex;
    state.tablePick = null;
    setSelection(mask, count, KD.levels[coloring.levels].labels[levelIndex]);
  }

  function selectByCluster(clusterLevel) {
    var codes = arr("cluster"), n = KD.n;
    if (!codes) return;
    if (state.tablePick === clusterLevel) { clearSelection(); return; }
    var mask = new Uint8Array(n), count = 0;
    for (var i = 0; i < n; i++) if (codes[i] === clusterLevel) { mask[i] = 1; count++; }
    state.tablePick = clusterLevel;
    state.legendPick = (state.coloring.levels === "cluster") ? clusterLevel : -1;
    setSelection(mask, count, KD.levels.cluster.labels[clusterLevel]);
  }

  // ---------------------------------------------------------------- legend

  var LEGEND_MAX = 40;

  function renderLegend() {
    var coloring = state.coloring;
    var list = $("kd-legend"), bar = $("kd-colorbar");
    $("kd-key-title").textContent = coloring.label;
    list.innerHTML = "";
    if (coloring.kind !== "cat") {
      list.hidden = true;
      bar.hidden = false;
      var stops = coloring.scale.map(function (s) {
        return s[1] + " " + (s[0] * 100).toFixed(0) + "%";
      }).join(", ");
      $("kd-colorbar-bar").style.background = "linear-gradient(90deg, " + stops + ")";
      $("kd-cb-lo").textContent = fmtNum(coloring.cmin, coloring.dp);
      $("kd-cb-mid").textContent = fmtNum((coloring.cmin + coloring.cmax) / 2, coloring.dp);
      $("kd-cb-hi").textContent = fmtNum(coloring.cmax, coloring.dp);
      $("kd-cb-unit").textContent = coloring.unit || "";
      return;
    }
    bar.hidden = true;
    list.hidden = false;
    var lv = KD.levels[coloring.levels];
    if (!lv) return;  // nothing has been computed yet (an empty run directory)
    var order = lv.order || lv.labels.map(function (_, i) { return i; });
    var shown = order.slice(0, LEGEND_MAX);
    shown.forEach(function (i) {
      var li = document.createElement("li");
      li.innerHTML = '<span class="kd-swatch" style="background:' + lv.colors[i] + '"></span>' +
        '<span class="lab" title="' + esc(lv.labels[i]) + '">' + esc(lv.labels[i]) + "</span>" +
        '<span class="cnt">' + fmtInt(lv.counts[i]) + "</span>";
      if (state.legendPick === i) li.className = "is-on";
      else if (state.legendPick >= 0) li.className = "is-muted";
      li.addEventListener("click", function () { selectByLevel(i); });
      list.appendChild(li);
    });
    if (order.length > LEGEND_MAX) {
      var more = document.createElement("li");
      more.className = "more";
      more.textContent = "+ " + (order.length - LEGEND_MAX) + " more";
      list.appendChild(more);
    }
  }

  // ---------------------------------------------------------------- tooltip

  var tip = $("kd-tooltip");

  function hoverHTML(p) {
    var L = KD.levels;
    var start = arr("start")[p];
    var span = KD.spans ? arr("span")[p] : KD.binSize;
    var chromLab = L.chrom.labels[arr("chrom")[p]] || "unplaced";
    var contigLab = L.contig ? L.contig.labels[arr("contig")[p]] : chromLab;
    var asm = L.assembly.labels[arr("assembly")[p]];
    var clusterIdx = arr("cluster") ? arr("cluster")[p] : -1;
    var featIdx = arr("feature") ? arr("feature")[p] : -1;
    var color = baseColors(state.coloring)[p];
    var html = '<div class="t-head"><span class="kd-swatch" style="background:' + color +
      '"></span>' + esc(chromLab) + ":" + fmtInt(start) + "–" + fmtInt(start + span) + "</div>";
    html += '<div class="t-row"><b>assembly</b> ' + esc(asm) + "</div>";
    if (clusterIdx >= 0) {
      html += '<div class="t-row"><b>cluster</b> <span class="kd-swatch" style="background:' +
        L.cluster.colors[clusterIdx] + ';display:inline-block;vertical-align:-1px"></span> ' +
        esc(L.cluster.labels[clusterIdx]) + "</div>";
    }
    if (featIdx >= 0) {
      html += '<div class="t-row"><b>feature</b> ' + esc(L.feature.labels[featIdx]) +
        " (" + fmtNum(arr("domfrac") ? arr("domfrac")[p] : 0, 2) + ")</div>";
    }
    html += '<div class="t-row"><b>gc</b> ' + fmtNum(arr("gc")[p], 3) +
      " · <b>sketch</b> " + fmtInt(Math.pow(10, arr("logns")[p])) + " hashes</div>";
    html += '<div class="t-uid">' + esc(asm + "|" + contigLab + "|" + start) + "</div>";
    return html;
  }

  function showTip(ev, gd, viaRibbon) {
    if (!ev || !ev.points || !ev.points.length) return;
    var idx = ev.points[0].pointIndex;
    if (idx === undefined) idx = ev.points[0].pointNumber;
    if (idx === undefined) return;
    var p = viaRibbon ? ribbonOrder[idx] : idx;
    if (!(p >= 0 && p < KD.n)) return;
    tip.innerHTML = hoverHTML(p);
    tip.classList.add("on");
    tip.setAttribute("aria-hidden", "false");
    var mouse = ev.event;
    var x = 0, y = 0;
    if (mouse && mouse.clientX !== undefined) { x = mouse.clientX; y = mouse.clientY; }
    else {
      var box = gd.getBoundingClientRect();
      x = box.left + box.width / 2; y = box.top + 20;
    }
    var w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(window.innerWidth - w - 10, Math.max(8, x + 16)) + "px";
    tip.style.top = Math.min(window.innerHeight - h - 10, Math.max(8, y - h - 12)) + "px";
  }

  function hideTip() {
    tip.classList.remove("on");
    tip.setAttribute("aria-hidden", "true");
  }

  // ---------------------------------------------------------------- table

  var sortKey = null, sortDir = -1;

  function renderTable() {
    var t = KD.table, el = $("kd-table");
    if (!t || !t.rows.length) {
      $("kd-table-card").querySelector(".kd-tablewrap").innerHTML =
        '<div class="kd-empty">No cluster table &mdash; run the <code>enrich</code> stage.</div>';
      $("kd-table-sub").textContent = "";
      return;
    }
    if (sortKey === null) { sortKey = "size"; sortDir = -1; }
    var cols = t.columns;
    var rows = t.rows.slice();
    var ci = cols.map(function (c) { return c.key; }).indexOf(sortKey);
    if (ci >= 0) {
      rows.sort(function (a, b) {
        var va = a[ci], vb = b[ci];
        if (va === null || va === undefined) return 1;
        if (vb === null || vb === undefined) return -1;
        if (typeof va === "string") return sortDir * va.localeCompare(vb);
        return sortDir * (va - vb);
      });
    }
    var maxSize = 0;
    var si = cols.map(function (c) { return c.key; }).indexOf("size");
    if (si >= 0) rows.forEach(function (r) { maxSize = Math.max(maxSize, r[si] || 0); });

    var html = "<thead><tr>";
    cols.forEach(function (c) {
      var on = c.key === sortKey;
      html += '<th data-key="' + esc(c.key) + '"' + (on ? ' class="sorted"' : "") + ">" +
        esc(c.label) + (on ? ' <span class="arrow">' + (sortDir < 0 ? "▼" : "▲") +
          "</span>" : "") + "</th>";
    });
    html += "</tr></thead><tbody>";
    rows.forEach(function (r) {
      var lvl = r[r.length - 1];
      html += '<tr data-level="' + lvl + '">';
      cols.forEach(function (c, k) {
        var v = r[k];
        if (c.type === "name") {
          html += '<td class="name"><span class="kd-swatch" style="background:' +
            (KD.levels.cluster.colors[lvl] || KD.missingColor) + '"></span>' + esc(v) + "</td>";
        } else if (c.type === "text") {
          html += '<td class="feat" title="' + esc(v === null ? "" : v) + '">' +
            esc(v === null ? "–" : v) + "</td>";
        } else if (c.type === "int") {
          var bar = (c.key === "size" && maxSize > 0)
            ? '<span class="kd-bar" style="width:' + (36 * v / maxSize).toFixed(1) + 'px"></span>'
            : "";
          html += "<td>" + fmtInt(v) + bar + "</td>";
        } else if (c.type === "pct") {
          html += "<td>" + (v === null || v === undefined ? "–" :
            (100 * v).toFixed(0) + "%") + "</td>";
        } else {
          html += "<td>" + fmtNum(v, 2) + "</td>";
        }
      });
      html += "</tr>";
    });
    html += "</tbody>";
    el.innerHTML = html;
    Array.prototype.forEach.call(el.querySelectorAll("th"), function (th) {
      th.addEventListener("click", function () {
        var key = th.getAttribute("data-key");
        if (key === sortKey) sortDir = -sortDir;
        else { sortKey = key; sortDir = (key === "name" || key === "top_features") ? 1 : -1; }
        renderTable();
        markTableRows();
      });
    });
    Array.prototype.forEach.call(el.querySelectorAll("tbody tr"), function (tr) {
      tr.addEventListener("click", function () {
        selectByCluster(parseInt(tr.getAttribute("data-level"), 10));
      });
    });
    $("kd-table-sub").textContent = t.rows.length + " clusters";
    markTableRows();
  }

  function markTableRows() {
    var el = $("kd-table");
    if (!el) return;
    Array.prototype.forEach.call(el.querySelectorAll("tbody tr"), function (tr) {
      var lvl = parseInt(tr.getAttribute("data-level"), 10);
      tr.classList.toggle("is-on", state.tablePick === lvl);
    });
  }

  // ---------------------------------------------------------------- theme

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    $("kd-theme").textContent = theme === "light" ? "Dark" : "Light";
    try { window.localStorage.setItem("kmer-dust-theme", theme); } catch (err) { /* file:// */ }
    if (!HAS_PLOTLY) return;
    var relayout = {
      font: plotFont(),
      "xaxis.gridcolor": cssVar("--grid"),
      "yaxis.gridcolor": cssVar("--grid"),
      "xaxis.tickfont.color": cssVar("--muted"),
      "yaxis.tickfont.color": cssVar("--muted")
    };
    if (plotted(umapDiv)) Plotly.relayout(umapDiv, relayout);
    if (plotted(ribbonDiv)) {
      var shapeUpdate = {};
      ribbon.rows.forEach(function (_, i) { shapeUpdate["shapes[" + i + "].fillcolor"] = cssVar("--dim"); });
      shapeUpdate["yaxis.tickfont.color"] = cssVar("--ink-2");
      shapeUpdate["xaxis.gridcolor"] = cssVar("--grid");
      shapeUpdate.font = plotFont();
      Plotly.relayout(ribbonDiv, shapeUpdate);
    }
    if (plotted(heatDiv)) {
      Plotly.restyle(heatDiv, {
        colorscale: [isDark() ? KD.heatmap.scales.dark : KD.heatmap.scales.light]
      }, [0]);
      Plotly.relayout(heatDiv, {
        font: plotFont(),
        "xaxis.tickfont.color": cssVar("--muted"),
        "yaxis.tickfont.color": cssVar("--ink-2")
      });
    }
    repaint();
  }

  // ---------------------------------------------------------------- boot

  function boot() {
    var stored = null;
    try { stored = window.localStorage.getItem("kmer-dust-theme"); } catch (err) { stored = null; }
    // Precedence: this reader's own toggle, then whatever the host already
    // stamped on <html>, then dark.  The host case matters when the report is
    // embedded rather than opened as its own document -- overwriting a theme
    // the surrounding page had already chosen is rude, and the palette is
    // complete in both directions either way.
    var preset = document.documentElement.getAttribute("data-theme");
    var theme = stored === "light" || stored === "dark" ? stored
      : (preset === "light" || preset === "dark" ? preset : "dark");
    document.documentElement.setAttribute("data-theme", theme);
    $("kd-theme").textContent = theme === "light" ? "Dark" : "Light";
    $("kd-theme").addEventListener("click", function () {
      applyTheme(isDark() ? "light" : "dark");
    });
    $("kd-clear").addEventListener("click", clearSelection);

    var colorings = KD.colorings || [];
    if (!colorings.length) {
      colorings = [{ key: "none", label: "nothing to colour", kind: "cat", array: "source",
        levels: "source" }];
    }
    state.coloring = colorings[0];
    var sel = $("kd-colorby");
    colorings.forEach(function (c, i) {
      var opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = c.label;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", function () {
      state.coloring = colorings[parseInt(sel.value, 10)] || colorings[0];
      state.legendPick = -1;
      _current = null;
      repaint();
      renderLegend();
    });

    $("kd-umap-count").textContent = KD.n
      ? fmtInt(KD.n) + " of " + fmtInt(KD.nTotal) + " bins shown"
      : "no bins";

    buildUmap();
    buildRibbon();
    buildHeatmap();
    renderLegend();
    renderTable();

    if (plotted(ribbonDiv)) {
      var resizeTimer = null;
      window.addEventListener("resize", function () {
        window.clearTimeout(resizeTimer);
        resizeTimer = window.setTimeout(function () {
          Plotly.restyle(ribbonDiv, { "marker.size": ribbonMarkerSize() }, [0]);
        }, 180);
      });
    }

    // Exposed for debugging and for the package's own smoke test.
    window.__kd = {
      state: state, arr: arr, hoverHTML: hoverHTML, repaint: repaint,
      selectByLevel: selectByLevel, selectByCluster: selectByCluster,
      clearSelection: clearSelection, applyTheme: applyTheme, ribbonClick: onRibbonClick
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
