const elements = {
  refreshButton: document.getElementById("refreshButton"),
  modeChip: document.getElementById("modeChip"),
  unitChip: document.getElementById("unitChip"),
  updatedChip: document.getElementById("updatedChip"),
  equityValue: document.getElementById("equityValue"),
  totalPnlValue: document.getElementById("totalPnlValue"),
  unrealizedPnlValue: document.getElementById("unrealizedPnlValue"),
  realizedPnlValue: document.getElementById("realizedPnlValue"),
  cashValue: document.getElementById("cashValue"),
  returnRateValue: document.getElementById("returnRateValue"),
  historyCount: document.getElementById("historyCount"),
  chartStartLabel: document.getElementById("chartStartLabel"),
  chartRangeLabel: document.getElementById("chartRangeLabel"),
  chartEndLabel: document.getElementById("chartEndLabel"),
  positionCountLabel: document.getElementById("positionCountLabel"),
  positionsTableBody: document.getElementById("positionsTableBody"),
  chart: document.getElementById("equityChart"),
  areaPath: document.getElementById("areaPath"),
  lineGlowPath: document.getElementById("lineGlowPath"),
  linePath: document.getElementById("linePath"),
  gridLayer: document.querySelector("#equityChart .grid"),
  axisLayer: document.getElementById("axisLayer"),
  yLabelLayer: document.getElementById("yLabelLayer"),
  xLabelLayer: document.getElementById("xLabelLayer"),
  intervalControls: document.getElementById("intervalControls"),
};

const locale = "en-US";
const chartDimensions = {
  width: 1200,
  height: 460,
  padding: { top: 28, right: 34, bottom: 84, left: 96 },
};
let currentInterval = "raw";

function formatMoney(value, unit) {
  return `${new Intl.NumberFormat(locale, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  }).format(Number(value || 0))} ${unit}`;
}

function formatPct(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function formatCompactDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value || "");
  }
  return new Intl.DateTimeFormat(locale, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatChartDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value || "");
  }
  return new Intl.DateTimeFormat(locale, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatAxisValue(value) {
  const absolute = Math.abs(Number(value || 0));
  const maximumFractionDigits = absolute >= 1000 ? 2 : 4;
  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  }).format(Number(value || 0));
}

function formatTimeLabel(value, options) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value || "");
  }
  return new Intl.DateTimeFormat(locale, options).format(date);
}

function formatXAxisLabelParts(value, interval) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return [String(value || "")];
  }

  if (interval === "day") {
    return [
      formatTimeLabel(date, {
        month: "short",
        day: "2-digit",
      }),
    ];
  }

  const dayPart = formatTimeLabel(date, {
    month: "short",
    day: "2-digit",
  });
  const timePart = formatTimeLabel(date, {
    hour: "2-digit",
    minute: interval === "raw" ? "2-digit" : undefined,
    hour12: false,
  });

  return [dayPart, interval === "hour" ? `${timePart}:00` : timePart];
}

function setSignedClass(node, value) {
  node.classList.remove("is-positive", "is-negative");
  if (value > 0) {
    node.classList.add("is-positive");
  } else if (value < 0) {
    node.classList.add("is-negative");
  }
}

function animateValue(node, target, formatter, options = {}) {
  const start = Number(node.dataset.value || 0);
  const end = Number(target || 0);
  const duration = options.duration || 900;
  const startTs = performance.now();

  function tick(now) {
    const progress = Math.min((now - startTs) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = start + (end - start) * eased;
    node.textContent = formatter(current);
    if (progress < 1) {
      requestAnimationFrame(tick);
    } else {
      node.dataset.value = String(end);
      node.textContent = formatter(end);
    }
  }

  requestAnimationFrame(tick);
}

function renderSummary(payload) {
  const unit = payload.pnl_unit || "USDT";
  const summary = payload.summary || {};
  const syncError = String(payload.portfolio_sync_error || "").trim();
  const portfolioSource = String(payload.portfolio_source || "").trim();
  const syncState = syncError ? "sync stale" : (portfolioSource === "okx_live" ? "live" : "local");

  elements.modeChip.textContent = `Mode ${String(payload.execution_mode || "").toUpperCase()}`;
  elements.unitChip.textContent = `Unit ${unit}`;
  elements.updatedChip.textContent = `Updated ${formatCompactDate(summary.updated_at)} | ${syncState}`;

  animateValue(elements.equityValue, summary.total_equity, (value) => formatMoney(value, unit));
  animateValue(elements.totalPnlValue, summary.total_pnl, (value) => formatMoney(value, unit));
  animateValue(elements.unrealizedPnlValue, summary.unrealized_pnl, (value) => formatMoney(value, unit));
  animateValue(elements.realizedPnlValue, summary.realized_pnl, (value) => formatMoney(value, unit));
  animateValue(elements.cashValue, summary.cash, (value) => formatMoney(value, unit));
  animateValue(elements.returnRateValue, summary.return_rate, (value) => formatPct(value));

  setSignedClass(elements.totalPnlValue, summary.total_pnl);
  setSignedClass(elements.unrealizedPnlValue, summary.unrealized_pnl);
  setSignedClass(elements.realizedPnlValue, summary.realized_pnl);
  setSignedClass(elements.returnRateValue, summary.return_rate);

  elements.positionCountLabel.textContent = `${summary.position_count || 0} position${summary.position_count === 1 ? "" : "s"}`;
}

function renderIntervalControls(payload) {
  const options = Array.isArray(payload.history_interval_options) ? payload.history_interval_options : [];
  const active = payload.history_interval || currentInterval;
  currentInterval = active;

  elements.intervalControls.innerHTML = options.map((option) => {
    const activeClass = option.value === active ? "is-active" : "";
    return `<button class="interval-button ${activeClass}" type="button" data-interval="${option.value}">${option.label}</button>`;
  }).join("");

  elements.intervalControls.querySelectorAll("[data-interval]").forEach((button) => {
    button.addEventListener("click", () => {
      const nextInterval = button.getAttribute("data-interval") || "raw";
      if (nextInterval === currentInterval) {
        return;
      }
      currentInterval = nextInterval;
      loadDashboard();
    });
  });
}

function renderPositions(payload) {
  const unit = payload.pnl_unit || "USDT";
  const positions = Array.isArray(payload.positions) ? payload.positions : [];

  if (!positions.length) {
    elements.positionsTableBody.innerHTML = `
      <tr>
        <td colspan="7" class="empty-state">No open positions right now.</td>
      </tr>
    `;
    return;
  }

  elements.positionsTableBody.innerHTML = positions.map((position) => {
    const unrealClass = Number(position.unrealized_pnl) >= 0 ? "is-positive" : "is-negative";
    const returnClass = Number(position.return_pct) >= 0 ? "is-positive" : "is-negative";
    return `
      <tr>
        <td>${position.symbol}</td>
        <td>${Number(position.quantity || 0).toFixed(8)}</td>
        <td>${formatMoney(position.avg_entry_price, unit)}</td>
        <td>${formatMoney(position.market_price, unit)}</td>
        <td>${formatMoney(position.market_value, unit)}</td>
        <td class="${unrealClass}">${formatMoney(position.unrealized_pnl, unit)}</td>
        <td class="${returnClass}">${formatPct(position.return_pct)}</td>
      </tr>
    `;
  }).join("");
}

function buildNiceStep(value) {
  if (!Number.isFinite(value) || value <= 0) {
    return 1;
  }

  const exponent = Math.floor(Math.log10(value));
  const fraction = value / (10 ** exponent);

  if (fraction <= 1) {
    return 10 ** exponent;
  }
  if (fraction <= 2) {
    return 2 * (10 ** exponent);
  }
  if (fraction <= 5) {
    return 5 * (10 ** exponent);
  }
  return 10 * (10 ** exponent);
}

function buildYTicks(minValue, maxValue, preferredCount = 5) {
  const range = Math.max(maxValue - minValue, 1);
  const step = buildNiceStep(range / Math.max(preferredCount - 1, 1));
  const tickMin = Math.floor(minValue / step) * step;
  const tickMax = Math.ceil(maxValue / step) * step;
  const ticks = [];

  for (let value = tickMin; value <= tickMax + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }

  if (ticks.length < 2) {
    ticks.push(Number((tickMin + step).toFixed(10)));
  }
  return ticks;
}

function selectTickIndexes(count, maxLabels = 6) {
  if (count <= 0) {
    return [];
  }
  if (count <= maxLabels) {
    return Array.from({ length: count }, (_, index) => index);
  }

  const step = (count - 1) / (maxLabels - 1);
  const indexes = new Set([0, count - 1]);
  for (let cursor = 1; cursor < maxLabels - 1; cursor += 1) {
    indexes.add(Math.round(cursor * step));
  }
  return Array.from(indexes).sort((left, right) => left - right);
}

function buildChartGeometry(history, width, height, padding) {
  const source = Array.isArray(history) && history.length
    ? history
    : [{ timestamp: new Date().toISOString(), equity: 0 }];
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const actualValues = source.map((item) => Number(item.equity || 0));
  const minValue = Math.min(...actualValues);
  const maxValue = Math.max(...actualValues);
  const spread = Math.max(maxValue - minValue, Math.max(Math.abs(maxValue), 1) * 0.04, 1);
  const provisionalMin = minValue - spread * 0.18;
  const provisionalMax = maxValue + spread * 0.18;
  const yTicks = buildYTicks(provisionalMin, provisionalMax);
  const domainMin = yTicks[0];
  const domainMax = yTicks.at(-1) ?? provisionalMax;
  const usable = source.length === 1 ? [source[0], source[0]] : source;

  const toY = (value) => {
    return padding.top + (1 - (Number(value || 0) - domainMin) / Math.max(domainMax - domainMin, 1)) * innerHeight;
  };

  const plotPoints = usable.map((item, index) => {
    const x = padding.left + (innerWidth * index) / Math.max(usable.length - 1, 1);
    return { x, y: toY(item.equity), equity: Number(item.equity || 0), timestamp: item.timestamp };
  });

  const actualPoints = source.length === 1
    ? [{
      x: padding.left + innerWidth / 2,
      y: toY(source[0].equity),
      equity: Number(source[0].equity || 0),
      timestamp: source[0].timestamp,
    }]
    : source.map((item, index) => {
      const x = padding.left + (innerWidth * index) / Math.max(source.length - 1, 1);
      return { x, y: toY(item.equity), equity: Number(item.equity || 0), timestamp: item.timestamp };
    });

  const xTickIndexes = selectTickIndexes(actualPoints.length, 6);
  const xTicks = xTickIndexes.map((index) => actualPoints[index]).filter(Boolean);

  return {
    domainMin,
    domainMax,
    plotPoints,
    actualPoints,
    xTicks,
    yTicks,
    innerHeight,
    innerWidth,
    toY,
  };
}

function renderGrid(width, height, padding, geometry) {
  const lines = [];
  const chartRight = width - padding.right;
  const chartBottom = height - padding.bottom;

  geometry.yTicks.forEach((value) => {
    const y = geometry.toY(value);
    lines.push(`<line x1="${padding.left}" y1="${y}" x2="${chartRight}" y2="${y}"></line>`);
  });

  geometry.xTicks.forEach((point) => {
    lines.push(`<line x1="${point.x}" y1="${padding.top}" x2="${point.x}" y2="${chartBottom}"></line>`);
  });

  elements.gridLayer.innerHTML = lines.join("");
}

function renderAxes(width, height, padding, geometry, interval) {
  const chartRight = width - padding.right;
  const chartBottom = height - padding.bottom;
  const axisLines = [
    `<line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${chartBottom}"></line>`,
    `<line x1="${padding.left}" y1="${chartBottom}" x2="${chartRight}" y2="${chartBottom}"></line>`,
  ];

  geometry.yTicks.forEach((value) => {
    const y = geometry.toY(value);
    axisLines.push(`<line x1="${padding.left - 8}" y1="${y}" x2="${padding.left}" y2="${y}"></line>`);
  });

  geometry.xTicks.forEach((point) => {
    axisLines.push(`<line x1="${point.x}" y1="${chartBottom}" x2="${point.x}" y2="${chartBottom + 8}"></line>`);
  });

  elements.axisLayer.innerHTML = axisLines.join("");

  elements.yLabelLayer.innerHTML = geometry.yTicks.map((value) => {
    const y = geometry.toY(value);
    return `<text x="${padding.left - 14}" y="${y + 5}">${formatAxisValue(value)}</text>`;
  }).join("");

  elements.xLabelLayer.innerHTML = geometry.xTicks.map((point) => {
    const labelParts = formatXAxisLabelParts(point.timestamp, interval);
    const x = point.x;
    const y = chartBottom + 22;

    if (labelParts.length === 1) {
      return `<text x="${x}" y="${y}">${labelParts[0]}</text>`;
    }

    return `
      <text x="${x}" y="${y}">
        <tspan x="${x}" dy="0">${labelParts[0]}</tspan>
        <tspan x="${x}" dy="16">${labelParts[1]}</tspan>
      </text>
    `;
  }).join("");
}

function pathFromPoints(points) {
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
}

function renderChart(payload) {
  const history = Array.isArray(payload.history) ? payload.history : [];
  const { width, height, padding } = chartDimensions;
  const geometry = buildChartGeometry(history, width, height, padding);
  const points = geometry.plotPoints;
  const interval = payload.history_interval || currentInterval;

  renderGrid(width, height, padding, geometry);
  renderAxes(width, height, padding, geometry, interval);

  const line = pathFromPoints(points);
  const baseY = height - padding.bottom;
  const area = `${line} L ${points[points.length - 1].x} ${baseY} L ${points[0].x} ${baseY} Z`;

  elements.areaPath.setAttribute("d", area);
  elements.lineGlowPath.setAttribute("d", line);
  elements.linePath.setAttribute("d", line);

  const length = elements.linePath.getTotalLength();
  elements.linePath.style.transition = "none";
  elements.lineGlowPath.style.transition = "none";
  elements.linePath.style.strokeDasharray = `${length}`;
  elements.lineGlowPath.style.strokeDasharray = `${length}`;
  elements.linePath.style.strokeDashoffset = `${length}`;
  elements.lineGlowPath.style.strokeDashoffset = `${length}`;
  elements.areaPath.style.opacity = "0.08";

  requestAnimationFrame(() => {
    elements.linePath.style.transition = "stroke-dashoffset 1.9s cubic-bezier(0.22, 1, 0.36, 1)";
    elements.lineGlowPath.style.transition = "stroke-dashoffset 1.9s cubic-bezier(0.22, 1, 0.36, 1)";
    elements.areaPath.style.transition = "opacity 1.2s ease";
    elements.linePath.style.strokeDashoffset = "0";
    elements.lineGlowPath.style.strokeDashoffset = "0";
    elements.areaPath.style.opacity = "0.26";
  });

  const historyLabel = payload.history_interval_label || "Raw";
  elements.historyCount.textContent = `${history.length || 1} ${historyLabel.toLowerCase()} point${history.length === 1 ? "" : "s"}`;

  const first = history[0] || history.at(-1);
  const last = history.at(-1) || history[0];
  const minEquity = Math.min(...history.map((item) => Number(item.equity || 0)));
  const maxEquity = Math.max(...history.map((item) => Number(item.equity || 0)));
  const unit = payload.pnl_unit || "USDT";

  elements.chartStartLabel.textContent = `Start ${formatChartDate(first?.timestamp)} | ${historyLabel}`;
  elements.chartEndLabel.textContent = `End ${formatChartDate(last?.timestamp)}`;
  elements.chartRangeLabel.textContent = `Range ${formatMoney(minEquity, unit)} to ${formatMoney(maxEquity, unit)}`;
}

async function loadDashboard() {
  elements.refreshButton.disabled = true;
  elements.refreshButton.textContent = "Refreshing";
  try {
    const response = await fetch(`/api/dashboard?interval=${encodeURIComponent(currentInterval)}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    const payload = await response.json();
    renderIntervalControls(payload);
    renderSummary(payload);
    renderPositions(payload);
    renderChart(payload);
  } catch (error) {
    console.error(error);
    elements.updatedChip.textContent = "Unable to load dashboard data";
  } finally {
    elements.refreshButton.disabled = false;
    elements.refreshButton.textContent = "Refresh";
  }
}

elements.refreshButton.addEventListener("click", loadDashboard);
loadDashboard();
setInterval(loadDashboard, 30000);
