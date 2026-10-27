"use strict";

((scope) => {
  function render() {
    const data = scope.SPILL_PERFORMANCE_SUMMARY;
    const body = document.querySelector("#spill-performance-body");
    if (!data || !body) return;
    const labels = {A5_historical: "A5 历史 35 项", A5_current: data.a5_reoptimized ? "A5 本轮寻优＋CAModel 验证" : "A5 同模型复算", A6: data.a6_compiler_checked ? "A6 寻优＋VF 编译期无 spill 检查" : "A6 本轮约束寻优"};
    const arch = document.querySelector("#arch-filter").value;
    body.replaceChildren();
    for (const dataset of arch === "A6" ? ["A6"] : ["A5_historical", "A5_current"]) {
      for (const group of data.statistics[dataset]) {
        const tr = document.createElement("tr");
        tr.dataset.scope = dataset;
        tr.dataset.kind = group.kind;
        const values = [labels[dataset], group.label, group.scope, group.valid, group.missing,
          ...group.counts.map(n => `${n} (${group.valid ? (100 * n / group.valid).toFixed(1) : "0.0"}%)`)];
        values.forEach((value, i) => {
          const td = document.createElement(i === 1 ? "th" : "td");
          if (i === 1) td.scope = "row";
          td.textContent = String(value);
          if (i >= 2) td.className = "num";
          tr.append(td);
        });
        body.append(tr);
      }
    }
    const missing = Object.values(data.rows).filter(r => r[arch === "A6" ? "A6" : "A5_current"].ratio == null);
    document.querySelector("#spill-performance-missing").textContent = missing.length
      ? `暂不可比较（不计入分母）：${missing.map(r => r.id).join("、")}。详细诊断见寻优说明。` : "35 项均可比较。";
    const approximate = Object.values(data.rows).filter(r => r[arch === "A6" ? "A6" : "A5_current"].scalar_spill_excluded);
    document.querySelector("#spill-performance-scalar-note").textContent = approximate.length
      ? `本次按用户确认口径，${approximate.length} 项忽略标量 spill 时序、只预测向量/谓词，已计入有效分母：${approximate.map(r => r.id).join("、")}。地址和向量/谓词搬运、屏障仍按原始日志保留。` : "";
  }
  scope.SpillPerformanceView = {render};
  if (typeof document !== "undefined") {
    document.querySelector("#arch-filter")?.addEventListener("change", render);
    render();
  }
})(typeof window === "undefined" ? globalThis : window);
