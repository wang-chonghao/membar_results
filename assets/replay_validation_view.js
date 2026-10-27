"use strict";

(() => {
  const audit = window.REPLAY_VALIDATION;
  if (!audit) return;
  const labels = {strict_pass: "A5 回放：验收通过（RV 时序也一致）", equivalent_pass: "A5 回放：同周期验收通过",
    dataflow_cycle_only: "A5 回放：总周期一致，待复核",
    mismatch: "A5 回放：不一致", not_verified: "A5 回放：未通过验证", pending: "A5 回放：待验证", no_listing: "无回放文件"};
  function decorate() {
    for (const row of document.querySelectorAll("#results-body tr[data-id]")) {
      if (row.dataset.replayDecorated) continue;
      row.dataset.replayDecorated = "true";
      const record = audit.rows[row.dataset.id];
      if (!record || record.status === "no_listing") continue;
      const cell = row.querySelector(":scope > :first-child");
      const badge = document.createElement(record.comparison_path ? "a" : "span");
      badge.className = "replay-validation " + record.status;
      badge.textContent = labels[record.status];
      if (record.comparison_path) badge.href = record.comparison_path;
      badge.title = record.compiler_diagnostic || record.run_diagnostic || record.recheck_error || record.error || `归档 / 原版重跑 / 回放：${record.original_cycles ?? "--"} / ${record.baseline_cycles ?? "--"} / ${record.replay_cycles ?? "--"} cycles`;
      const link = Array.from(cell.querySelectorAll("a")).find(a => /(?:compiled_vf_replay|compiled_pc_replay|hardware_equivalent)\.cce$/.test(a.getAttribute("href")));
      if (link) {
        if (["strict_pass", "equivalent_pass"].includes(record.status) && record.source_path) {
          link.href = record.source_path;
          link.textContent = "已重编译验证的回放 CCE";
        } else {
          link.textContent = record.status === "mismatch" ? "回放审计清单（重编译不一致）" : "回放审计清单（未验收）";
        }
      }
      cell.append(badge);
    }
  }
  new MutationObserver(decorate).observe(document.querySelector("#results-body"), {childList: true});
  decorate();
})();
