"use strict";

(() => {
  const data = window.SPILL_REPORT;
  const tbody = document.querySelector("#results-body");
  const nodes = new Map(Array.from(tbody.rows, row => [row.dataset.id, row]));
  const filters = ["#case-filter", "#status-filter", "#search", "#origin-filter"].map(id => document.querySelector(id));
  let sortKey = null;
  let descending = false;

  function matches(row) {
    const [operator, state, query, origin] = filters.map(input => input.value);
    if (operator !== "all" && operator !== row.case) return false;
    if (origin !== "all" && origin !== row.origin) return false;
    if (![row.case,row.title,row.variant_label,row.comparison_label].join(" ").toLowerCase().includes(query.trim().toLowerCase())) return false;
    if (state === "spill") return row.spill && row.status !== "vf_stack_exceeded";
    if (state === "no-spill") return !row.spill;
    if (state === "unsupported") return row.status === "unsupported";
    return true;
  }

  function render() {
    const rows = data.rows.filter(matches);
    rows.sort((left, right) => {
      if (sortKey) {
        const a = left[sortKey], b = right[sortKey];
        if (a == null && b != null) return 1;
        if (b == null && a != null) return -1;
        const order = typeof a === "string" ? a.localeCompare(b) : a == null ? 0 : a - b;
        if (order) return descending ? -order : order;
      }
      return data.case_order.indexOf(left.case) - data.case_order.indexOf(right.case) || left.variant_order - right.variant_order;
    });
    tbody.replaceChildren(...rows.map(row => nodes.get(row.id)));
    document.querySelector("#row-count").textContent = `${rows.length} / ${data.rows.length} 个版本`;
    document.querySelector("#empty-state").hidden = rows.length !== 0;
  }
  filters.forEach(input => input.addEventListener("input", render));
  document.querySelectorAll("[data-sort]").forEach(button => button.addEventListener("click", () => {
    descending = sortKey === button.dataset.sort ? !descending : false;
    sortKey = button.dataset.sort;
    document.querySelectorAll("thead th").forEach(th => th.removeAttribute("aria-sort"));
    button.parentElement.setAttribute("aria-sort", descending ? "descending" : "ascending");
    render();
  }));

  const dialog = document.querySelector("#detail-dialog");
  const detail = document.querySelector("#detail-body");
  function paragraph(value, className) {
    const p = document.createElement("p");
    p.textContent = value;
    if (className) p.className = className;
    detail.append(p);
  }
  function files(title, entries, expanded = false) {
    if (!entries?.length) return;
    const group = document.createElement("details");
    group.open = expanded;
    const summary = document.createElement("summary");
    summary.textContent = `${title}（${entries.length}）`;
    group.append(summary);
    const list = document.createElement("ul");
    list.className = "file-list";
    entries.forEach(file => {
      const li = document.createElement("li"), link = document.createElement("a");
      link.href = file.path;
      link.textContent = file.label;
      li.append(link);
      list.append(li);
    });
    group.append(list);
    detail.append(group);
  }
  tbody.addEventListener("click", event => {
    const button = event.target.closest(".detail-button");
    if (!button) return;
    const row = data.rows.find(item => item.id === button.dataset.id);
    document.querySelector("#detail-title").textContent = `${row.title} · ${row.variant_label}`;
    detail.replaceChildren();
    const metrics = document.createElement("div");
    metrics.className = "detail-metrics";
    [["原 CAModel", row.camodel], ["VfSim 全局", row.global_cycles], ["VfSim 局部", row.local_cycles],
      ["VfSim 软件优化全局", row.software_cycles], ["优化后 CAModel", row.software_camodel]].forEach(([name, value]) => {
      const group = document.createElement("div"), label = document.createElement("span"), number = document.createElement("strong");
      label.textContent = name;
      number.textContent = value == null ? "--" : `${value} cycles`;
      group.append(label, number);
      metrics.append(group);
    });
    detail.append(metrics);
    paragraph(`${row.shape}；来源：${row.origin === "round3_search" ? "第三批新入口实测（尚未软件寻优）" : row.origin === "historical_membar" ? "历史 Membar 归档（未复测）" : "本轮固定 unroll 寻优"}。`);
    paragraph(`循环次数：${row.loop_count_label ?? "--"}。${row.loop_count_detail}`);
    paragraph(`Vector spill：写 ${row.vector_spill_store ?? "--"} / 读 ${row.vector_spill_load ?? "--"} 次；Predicate spill：写 ${row.predicate_spill_store ?? "--"} / 读 ${row.predicate_spill_load ?? "--"} 次。统计整个 VF 的动态搬运次数，不是寄存器个数。`);
    paragraph(row.spill_count_basis);
    if(row.other_extra_vector_loads)paragraph(`另有 ${row.other_extra_vector_loads} 次未配对额外向量读取，未作为 Vector spill 重载计数。`);
    if(row.scalar_stack_stores||row.scalar_stack_loads)paragraph(`另有标量栈访存：写 ${row.scalar_stack_stores ?? "--"} / 读 ${row.scalar_stack_loads ?? "--"} 次；不归入 Vector/Predicate spill。`);
    if (row.membar_vs_software_performance != null) {
      paragraph(`Membar / 软件性能：${row.membar_vs_software_performance.toFixed(2)}%（VfSim 软件优化 ${row.software_cycles} / VfSim 局部依赖 ${row.local_cycles}）。100% 表示持平，大于 100% 表示 Membar 优化更快。`);
    }
    if (row.reason) paragraph(row.reason, "reason");
    if (row.plan) {
      const plan = row.plan;
      paragraph(`${row.strategy}；${plan.stages} 段；cohort=${plan.cohort}；压力权重=${plan.pressure_weight}；临时 UB=${plan.scratch_bytes}B；UB 高水位=${plan.ub_high_water}B。`);
      paragraph(`保持原 U${row.unroll} 和算术；本轮 ${row.candidate_count} 个候选、${row.finalists} 个编译验证结果。AscendC/U1 逐位一致，独立 golden 与 guard 通过，无编译器 stack spill。`);
    } else if (row.software_comparison) {
      paragraph(`${row.strategy}：${row.software_comparison.case} / ${row.software_comparison.label}，保持 U${row.unroll}，使用该版本原 CAModel 和 VfSim 全局结果。`);
    } else {
      paragraph(row.strategy);
      if (row.status === "predicted" && !row.spill) paragraph("软件侧的 VfSim、CAModel 和精度均沿用原版本，没有重新寻优或实测。");
    }
    if (row.note) paragraph(row.note,"reason");
    if (row.source_cycles != null && row.spill) paragraph(`历史源码预测 ${row.source_cycles} cycles 不包含完整 compiler spill，未填入软件优化列。`);
    if (row.case_notes?.length) paragraph(row.case_notes.join("\n"));
    const links = [...(row.files.original_cce || []), ...(row.files.software || [])].filter(file => file.path.endsWith(".cce"));
    files("原始与软件优化 CCE", links, true);
    files("原始 CAModel 用例与日志", row.files.original_cce);
    files("原始全局 / 局部预测与地址审计", row.files.comparison);
    if (row.pure_cce_refresh) {
      const previous = row.previous_cce_measurements;
      paragraph(`本行已重新验证纯 CCE。转换前记录：CAModel ${previous.camodel}，VfSim 全局 ${previous.global_cycles}，局部 ${previous.local_cycles} cycles；仅作为历史证据保留。`);
      files("转换前源码与 CAModel 日志（历史，不用于当前预测）", row.files.previous_cce);
      files("转换前预测与地址审计（历史）", row.files.previous_comparison);
    }
    files("软件优化验证、Canonical 与预测日志", row.files.software);
    files("Spill 分类与数量证据",row.files.spill_audit);
    if(row.origin === "round3_search") files("原 AscendC、golden 与输入布局",row.files.reference);
    files("本行结果记录", [{label: "result.json", path: row.evidence_path}]);
    dialog.showModal();
  });
  document.querySelector("#close-detail").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    const box = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) dialog.close();
  });

  let chart;
  function plot(resetGroup = false) {
    const operator = document.querySelector("#chart-case").value;
    const candidates = data.rows.filter(row => row.case === operator).sort((a, b) => a.variant_order - b.variant_order);
    const groupSelect = document.querySelector("#chart-group");
    if (resetGroup || groupSelect.options.length === 0) {
      groupSelect.replaceChildren(...Array.from(new Map(candidates.map(row => [row.comparison_group,row.comparison_label])),([value,label]) => {
        const option = document.createElement("option");
        option.value = value; option.textContent = label; return option;
      }));
    }
    const rows = candidates.filter(row => row.comparison_group === groupSelect.value);
    const datasets = [["camodel", "原 CAModel", "#344251"], ["global_cycles", "VfSim 全局", "#2376b8"],
      ["local_cycles", "VfSim 局部", "#18846d"], ["software_cycles", "VfSim 软件优化全局 / 保留原版本", "#a94f69"]]
      .map(([key, label, color]) => ({label, data: rows.map(row => row[key]), borderColor: color,
        backgroundColor: color, pointRadius: 3, borderWidth: 2, spanGaps: false, tension: 0}));
    if (chart) chart.destroy();
    chart = new Chart(document.querySelector("#cycles-chart"), {
      type: "line", data: {labels: rows.map(row => row.chart_label), datasets},
      options: {responsive: true, maintainAspectRatio: false, animation: false,
        interaction: {intersect: false, mode: "index"},
        plugins: {legend: {position: "bottom", labels: {font: {size: 12}, boxWidth: 18}}},
        scales: {y: {beginAtZero: true, title: {display: true, text: "VF cycles"}}, x: {ticks: {maxRotation: 30, font: {size: 11}},title: {display: true, text: "版本 / Unroll（同 shape、同参数组）"}}}}
    });
  }
  document.querySelector("#chart-case").addEventListener("change", () => plot(true));
  document.querySelector("#chart-group").addEventListener("change", () => plot());
  filters.forEach(input => input.addEventListener("input", () => {
    const visible=data.rows.filter(matches);
    if(!visible.length)return;
    const selector=document.querySelector("#chart-case");
    const previousGroup=document.querySelector("#chart-group").value;
    if(!visible.some(row=>row.case===selector.value))selector.value=visible[0].case;
    const groups=Array.from(new Set(visible.filter(row=>row.case===selector.value).map(row=>row.comparison_group)));
    plot(true);
    document.querySelector("#chart-group").value=groups.includes(previousGroup)?previousGroup:groups[0];
    plot();
  }));
  window.SPILL_CHART = () => chart;
  render();
  plot();
})();
