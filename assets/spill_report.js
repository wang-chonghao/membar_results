"use strict";

(() => {
  const data = window.SPILL_REPORT;
  const baseRows = data.rows;
  const refreshed = window.MULTISOC_REPORT;
  const archFilter = document.querySelector("#arch-filter");
  const versionScope = document.querySelector("#version-scope");
  const mainIds = new Set(data.default_row_ids || baseRows.map(row => row.id));
  let architecture = archFilter?.value || "A5";
  const tbody = document.querySelector("#results-body");
  const nodes = new Map(Array.from(tbody.rows, row => [row.dataset.id, row]));
  const filters = ["#case-filter", "#status-filter", "#search", "#origin-filter"].map(id => document.querySelector(id));
  let sortKey = null;
  let descending = false;

  function matches(row) {
    if (versionScope?.value !== "all" && !mainIds.has(row.id)) return false;
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

  function applyArchitecture() {
    architecture = archFilter.value;
    data.rows = baseRows.map(row => window.MultiSocView.project(row, architecture, refreshed));
    const percentages = new Set(["original_accuracy", "local_gain", "software_accuracy", "membar_vs_software_performance", "a6_vs_a5_local_performance"]);
    for (const row of data.rows) {
      const node = nodes.get(row.id);
      for (const cell of node.querySelectorAll("[data-metric]")) {
        const key = cell.dataset.metric, value = row[key];
        cell.textContent = value == null ? "--" : percentages.has(key) ? `${value.toFixed(2)}%` : String(value);
        cell.title = architecture === "A6" && key.includes("accuracy") ? "无 A6 CAModel 实测，不能计算同代预测精度" : "";
      }
      node.classList.toggle("prediction-unavailable", row.status !== "predicted");
      node.querySelector(".detail-button").title = row.reason || `${architecture} 运行证据`;
      const strategyCell = node.querySelector(".strategy");
      if (strategyCell) strategyCell.textContent = row.strategy;
      node.querySelectorAll(".software-cce-links").forEach(link => link.remove());
      if (row.software_cycles != null) {
        for (const file of row.software_cce_links) {
          const line = document.createElement("small"), link = document.createElement("a");
          line.className = "software-cce-links";
          link.className = "software-cce-link";
          link.href = file.path;
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = file.label;
          link.title = file.vf_compile_only ? "A6 VF 编译期寄存器 spill 检查通过；外围 DMA/AscendC 包装不在验证范围，未做运行或数值验证。周期仍是源级 VfSim 预测。" : file.compiler_verified ? "对应当前软件预测，已通过 A5 CAModel 验证" : "对应当前 A6 软件预测；未做 A6 编译或 CAModel 验证";
          if (file.source_kind === "selected_canonical_ssa_listing") link.title += "；由选中的 Canonical 还原，重新编译可能改变调度或 spill";
          line.append(link);
          node.cells[0].append(line);
        }
      }
      node.querySelector(".scalar-timing-note")?.remove();
      if (row.scalar_spill_excluded) {
        const note = document.createElement("small");
        note.className = "scalar-timing-note";
        note.textContent = "* 仅向量/谓词计时，标量 spill 忽略";
        node.querySelector(".status").append(note);
      }
    }
    const predicted = data.rows.filter(row => row.status === "predicted").length;
    document.querySelector("#architecture-note").textContent = `${architecture === "A6" ? "A6 / DV121" : "A5 / DV100"} · 同分支重跑 · ${predicted}/${data.rows.length} 个归档版本有原始全局预测 · 模型 ${refreshed?.model_commit?.slice(0, 7) || "未加载"}。CAModel 为 A5 实测参考。`;
    for (const [key, label] of [["camodel", "原 CAModel（A5）"], ["software_camodel", "优化后 CAModel（A5）"],
                               ["original_accuracy", architecture === "A6" ? "全局精度（无 A6 实测）" : "全局精度"],
                               ["software_accuracy", architecture === "A6" ? "优化后精度（无 A6 实测）" : "优化后精度"]]) {
      const button = document.querySelector(`[data-sort="${key}"]`);
      if (button) button.textContent = `${label} ↕`;
    }
    render();
    plot();
  }
  filters.forEach(input => input.addEventListener("input", render));
  versionScope?.addEventListener("change", render);
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
    paragraph(`${row.arch} / ${row.arch === "A6" ? "DV121" : "DV100"}；模型 MultiSoC-membar-refactor ${refreshed?.model_commit?.slice(0, 7)}；${row.shape}。`);
    if (row.arch === "A6") paragraph(row.a6_search?.compiler_verified ? "软件方案已通过 A6 VF 编译期寄存器 spill 检查。外围 DMA/AscendC 包装不在检查范围，未做 A6 数值验证；周期仍是配对源级 Canonical 的 VfSim 预测，不是编译后二进制回放。CAModel 数字均为 A5 实测参考，不计算 A6 精度。" : row.a6_search ? "原 CAModel 为 A5 实测参考。软件列是本轮 A6 资源约束寻优结果，没有对应 CAModel 实测，不计算精度。" : "CAModel 数字仍为 A5 实测参考，不代表 A6 实测，也不计算 A6 预测精度。本行不在历史 35 项寻优范围，软件仍为历史选定实现。");
    if (row.architecture_evidence) files("当前架构重跑证据", [{label: "A5/A6 输入、诊断与逐 VF 周期", path: row.architecture_evidence}], true);
    paragraph(`历史预测仅供追溯：全局 ${row.legacy_metrics.global_cycles ?? "--"}，局部 ${row.legacy_metrics.local_cycles ?? "--"}，软件全局 ${row.legacy_metrics.software_cycles ?? "--"} cycles。当前表格不使用历史数值填补缺项。`);
    if (row.software_reason) paragraph(`软件优化预测：${row.software_reason}`, "reason");
    paragraph(`循环次数：${row.loop_count_label ?? "--"}。${row.loop_count_detail}`);
    paragraph(`Vector spill：写 ${row.vector_spill_store ?? "--"} / 读 ${row.vector_spill_load ?? "--"} 次；Predicate spill：写 ${row.predicate_spill_store ?? "--"} / 读 ${row.predicate_spill_load ?? "--"} 次。统计整个 VF 的动态搬运次数，不是寄存器个数。`);
    paragraph(row.spill_count_basis);
    if(row.other_extra_vector_loads)paragraph(`另有 ${row.other_extra_vector_loads} 次未配对额外向量读取，未作为 Vector spill 重载计数。`);
    if(row.scalar_stack_stores||row.scalar_stack_loads)paragraph(`另有标量栈访存：写 ${row.scalar_stack_stores ?? "--"} / 读 ${row.scalar_stack_loads ?? "--"} 次；不归入 Vector/Predicate spill。`);
    if (row.membar_vs_software_performance != null) {
      paragraph(`Membar / 软件性能：${row.membar_vs_software_performance.toFixed(2)}%（VfSim 软件优化 ${row.software_cycles} / VfSim 局部依赖 ${row.local_cycles}）。100% 表示持平，大于 100% 表示 Membar 优化更快。`);
    }
    if (row.reason) paragraph(row.reason, "reason");
    if (row.scalar_spill_excluded && row.architecture_record?.original?.evidence) {
      const evidence = row.architecture_record.original.evidence;
      files("原始日志回放与标量排除审计", [{label: "逐 PC、地址和指令数量审计", path: evidence.path},
        {label: "回放 CCE 阅读参考（预测使用 Canonical）", path: evidence.cce}], true);
    }
    if (row.a6_search) {
      const search = row.a6_search;
      paragraph(`${search.unique_candidates == null ? "A6 沿用软件方案" : `A6 搜索：${search.feasible_candidates}/${search.unique_candidates} 个唯一候选满足约束并成功预测；峰值 Vector=${search.peaks?.Register ?? "--"} / 32，Predicate=${search.peaks?.PredicateRegister ?? "--"} / 8`}。保持名义 U${row.unroll}；${search.compiler_verified ? `VF 编译期无寄存器 spill 已检查，选中排名 ${search.selected_rank}；未做运行/数值验证` : "未做 A6 编译器验证"}。`);
      files("本轮 A6 Canonical、源码与编译期检查", search.files, true);
      paragraph("下列软件 CCE 和 CAModel 链接为历史 A5 方案证据，不是本轮新排列的实测记录。新方案以以上 Canonical 为准。");
    } else if (row.a5_search) {
      const search = row.a5_search;
      paragraph(`A5 当前模型重新寻优；选中 CCE 已重编译、与已通过 golden 的参考输出逐字节比较，并检查无新增编译器 spill。软件周期来自该 CCE 编译后日志的全局 Membar 预测，CAModel 为该版本实测，不沿用历史方案。`);
      files("本轮 A5 软件 CCE、CAModel 与编译后 Canonical", search.files, true);
      paragraph("下列历史软件链接仅作归档参考，不是本轮选中方案。");
    } else if (row.plan) {
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
    const datasetSpecs = [["global_cycles", `${architecture} VfSim 全局`, "#2376b8"],
      ["local_cycles", "VfSim 局部", "#18846d"], ["software_cycles", "VfSim 软件优化全局 / 保留原版本", "#a94f69"]]
    if (architecture === "A5") datasetSpecs.unshift(["camodel", "A5 CAModel 实测", "#344251"]);
    else datasetSpecs.push(["a5_local_cycles", "A5 VfSim 局部（对照）", "#68737d"]);
    const datasets = datasetSpecs
      .map(([key, label, color]) => ({label, data: rows.map(row => row[key]), borderColor: color,
        backgroundColor: color, borderDash: key === "a5_local_cycles" ? [5, 4] : [], pointRadius: 3, borderWidth: 2, spanGaps: false, tension: 0}));
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
  window.SPILL_VIEW = () => data.rows;
  archFilter?.addEventListener("change", applyArchitecture);
  applyArchitecture();
})();
