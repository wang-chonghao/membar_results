"use strict";

((scope) => {
  const percent = (predicted, measured) => predicted == null || !measured ? null : 100 * (1 - Math.abs(predicted - measured) / measured);
  const ratio = (numerator, denominator) => numerator == null || !denominator ? null : 100 * numerator / denominator;
  function project(row, arch, refreshed) {
    const pair = refreshed?.rows?.[row.id];
    const current = pair?.arches?.[arch];
    const original = current?.original;
    const software = current?.software;
    const scalarExcluded = original?.evidence?.scalar_spill_opcodes_excluded;
    const scalarNote = scalarExcluded ? "按用户指定口径仅预测向量/谓词；标量 spill 时序不计入，访存地址按原日志恢复。" : "";
    const predicted = original?.status === "predicted";
    const globalCycles = predicted ? original.global_cycles : null;
    const localCycles = predicted ? original.local_cycles : null;
    const summary = scope.SPILL_PERFORMANCE_SUMMARY;
    const search = arch === "A6" ? summary?.rows?.[row.id]?.A6 || summary?.a6_compiler_checks?.[row.id] : summary?.a5_reoptimized ? summary.rows?.[row.id]?.A5_current : null;
    const softwareCycles = search ? search.software_cycles : software?.status === "predicted" ? software.global_cycles : null;
    const softwareCamodel = search ? arch === "A5" ? search.software_camodel : null : row.software_camodel;
    return {...row, arch, legacy_metrics: {global_cycles: row.global_cycles, local_cycles: row.local_cycles, software_cycles: row.software_cycles},
      global_cycles: globalCycles, local_cycles: localCycles, software_cycles: softwareCycles,
      software_camodel: softwareCamodel,
      a6_search: arch === "A6" ? search : null,
      a5_search: arch === "A5" ? search : null,
      software_cce_links: scope.SOFTWARE_CCE_LINKS?.rows?.[row.id]?.[arch] || [],
      scalar_spill_excluded: scalarExcluded || null,
      strategy: search ? `${arch} ${arch === "A6" && !summary?.rows?.[row.id] ? "沿用软件方案" : "32/8 活跃值约束寻优"}（全局 Membar）${arch === "A5" ? "，CAModel 验证" : search.compiler_verified ? "，VF 编译期无 spill 已检" : ""}` : row.strategy,
      original_accuracy: arch === "A5" ? percent(globalCycles, row.camodel) : null,
      software_accuracy: arch === "A5" ? percent(softwareCycles, softwareCamodel) : null,
      local_gain: globalCycles && localCycles != null ? 100 * (1 - localCycles / globalCycles) : null,
      membar_vs_software_performance: ratio(softwareCycles, localCycles),
      a6_vs_a5_local_performance: ratio(pair?.arches?.A5?.original?.local_cycles, pair?.arches?.A6?.original?.local_cycles),
      a5_local_cycles: pair?.arches?.A5?.original?.local_cycles ?? null,
      a5_global_cycles: pair?.arches?.A5?.original?.global_cycles ?? null,
      status: predicted ? "predicted" : "unsupported",
      reason: predicted ? [original.local_reason, scalarNote].filter(Boolean).join(" ") : original?.reason || "本版本尚无该架构的有效重跑结果",
      architecture_record: current,
      architecture_evidence: pair?.path,
      software_reason: search ? ["predicted", "validated_no_spill", "validated_vf_no_spill"].includes(search.search_status) ? "" : search.reason || `未找到满足 ${arch} 32/8 上限的有效候选` : software?.status === "predicted" ? "" : software?.reason || "没有可用的软件优化预测"};
  }
  scope.MultiSocView = {project, ratio};
  if (typeof module !== "undefined") module.exports = scope.MultiSocView;
})(typeof window === "undefined" ? globalThis : window);
