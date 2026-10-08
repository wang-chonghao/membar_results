"""Collect existing A5 experiments and derive a portable report dataset."""
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "results/ub_address_dependency_experiment"
MANIFEST = []
COLORS = {"camodel": "#344251", "global": "#2376b8", "local": "#18846d", "source": "#bc6137"}
LABELS = {"camodel": "CAModel 实测", "global": "VfSim 全局同步", "local": "VfSim 局部依赖", "source": "VfSim 源码预测"}


def read(path):
    return json.loads(path.read_text())


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    MANIFEST.append({"path": str(target.relative_to(DEST)), "source": str(source.relative_to(ROOT)), "sha256": digest})
    return str(target.relative_to(DEST))


def compute_opcode(op):
    op = op.removeprefix("RV_")
    return op.startswith("V") and not op.startswith(("VLD", "VST", "VLOOP", "VAG"))


def curve(cycles, key, ipc=None):
    if ipc is None:
        counts = Counter(cycles)
        first = min(counts)
        events = [counts.get(t + first, 0) for t in range(max(counts) - first + 1)]
        running = 0
        ipc = []
        for t, count in enumerate(events):
            running += count
            if t >= 10:
                running -= events[t - 10]
            ipc.append(round(running / 10, 4))
        total = len(cycles)
    else:
        # Historical CSV already uses the same trailing ten-cycle window.
        first = next(i for i, y in enumerate(ipc) if y > 0)
        last = max(i for i, y in enumerate(ipc) if y > 0)
        ipc = ipc[first:last + 1]
        total = None
    return {"key": key, "label": LABELS[key], "color": COLORS[key], "samples": ipc, "events": total}


def camodel_logs(folder, target):
    done = folder / "core0.veccore0.instr_log.dump"
    popped = folder / "core0.veccore0.instr_popped_log.dump"
    links = [copy_file(p, target / "camodel" / p.name) for p in (popped, done)]
    complete = done.read_text()
    vf = list(re.finditer(r"\[info\]\s*\[(\d+)\].*?\bVF\s+addr:.*?vf_execute_time:\s*(\d+)", complete))
    assert len(vf) == 1, str(done)
    end, duration = map(int, vf[0].groups())
    start = end - duration
    pattern = re.compile(r"\[info\]\s*\[(\d+)\].*?\(ID:\s*(\d+)\)\s+(RV_\w+)")
    events = {}
    for match in pattern.finditer(complete):
        cy, iid, op = match.groups()
        if start <= int(cy) <= end and compute_opcode(op):
            events[int(iid)] = int(cy)
    assert events, str(done)
    return duration, links, curve(list(events.values()), "camodel")


def model_curve(folder, key):
    done = folder / "done_by_cycle.json"
    events = [json.loads(line) for line in done.read_text().splitlines() if line.strip()]
    cycles = [row["cy"] for row in events if compute_opcode(row["op"]) and row.get("op_class", "COMPUTE") == "COMPUTE"]
    assert cycles, str(done)
    return curve(cycles, key)


def make_variant(case_id, variant_id, label, source, logs, cam, source_cycles=None,
                 global_cycles=None, local_cycles=None, source_model=None,
                 global_model=None, local_model=None, replay=None, prediction=None,
                 spill=(0, 0), predicate_spill=(0, 0), membars=0, note="", evidence=(), curves=None):
    directory = DEST / case_id / variant_id
    source_links = [{"label": "完整 CCE", "path": copy_file(source, directory / "source/kernel.cce")}]
    for extra, name, title in ((replay, "compiled_pc_replay.cce", "编译后 PC 回放"),
                               (prediction, "prediction_input.cce", "预测输入副本")):
        if extra:
            source_links.append({"label": title, "path": copy_file(extra, directory / "source" / name)})
    actual, log_links, cam_curve = camodel_logs(logs, directory)
    assert actual == cam, (case_id, variant_id, actual, cam)
    series = curves or [cam_curve]
    if curves is None:
        for model, key in ((global_model, "global"), (local_model, "local"), (source_model, "source")):
            if model:
                series.append(model_curve(model, key))
    evidence_links = []
    for i, item in enumerate(evidence):
        evidence_links.append(copy_file(item, directory / "evidence" / f"{i + 1}_{item.name}"))
    v = {"id": variant_id, "label": label, "camodel": cam, "source_cycles": source_cycles,
         "global_cycles": global_cycles, "local_cycles": local_cycles,
         "accuracy": round(100 * (1 - abs(global_cycles - cam) / cam), 4) if global_cycles else None,
         "speedup": round(global_cycles / local_cycles, 4) if global_cycles and local_cycles else None,
         "vector_store": spill[0], "vector_load": spill[1], "predicate_store": predicate_spill[0],
         "predicate_load": predicate_spill[1], "membars": membars, "golden": "已通过",
         "note": note, "sources": source_links, "logs": log_links, "evidence": evidence_links, "series": series}
    (directory / "result.json").write_text(json.dumps(v, ensure_ascii=False, indent=2))
    return v


def collect():
    cases = []
    # The original Softmax archive has the matched kernel and both raw logs.
    soft = EXPERIMENTS / "softmax_round56_vdup_vsts/archive/before_reciprocal_vdup_removal"
    csv_rows = list(csv.DictReader((soft / "ipc/compute_completion_ipc_window10.csv").open()))
    series = [curve([], key, [float(row[column]) for row in csv_rows]) for key, column in
              (("camodel", "camodel_ipc"), ("global", "vfsim_global_membar_ipc"), ("local", "vfsim_local_ub_dependency_ipc"))]
    v = make_variant("softmax_three_loop", "round56", "三循环 · 8 × 1024", soft / "attention_softmax_round56_vdup_vsts.cce",
                     soft / "camodel", 654, global_cycles=638, local_cycles=524, membars=2,
                     note="同源码的全局/局部同步 A/B，精确地址覆盖率 100%。此处采用原始证据完整的版本。",
                     evidence=(soft / "summary.json", soft / "README.md"), curves=series)
    cases.append({"id": "softmax_three_loop", "title": "Softmax", "subtitle": "三循环阶段同步", "shape": "8 行 × 1024，FP32",
                  "default": "round56", "headline": "消除不同循环阶段之间的等待空泡，模型内加速 1.22x。",
                  "notes": ["局部依赖允许不同 row 的 max、sum 和 norm 阶段重叠，保留实际 UB 读写约束。",
                            "较新版本移除 reciprocal VDUP：已有 CAModel 644、全局 637、局部 523 记录；原始两份日志和源码已缺失，因此未拿它们配对本目录的旧日志。"],
                  "variants": [v]})

    poly = EXPERIMENTS / "gelu_poly_i96_unroll_spill_20260911"
    comparisons = {r["unroll"]: r for r in read(poly / "camodel_vfsim_comparison.json")}
    variants = []
    for n in (1, 2, 4, 6, 8, 12):
        folder = poly / f"u{n}"
        s = read(folder / "summary.json")
        cmp = comparisons[n]
        replay = folder / "compiled_replay"
        ab = read(replay / "ub_comparison/summary.json") if n == 12 else None
        variants.append(make_variant("gelu_poly", f"u{n}", f"Unroll {n}", folder / f"source/gelu_poly_i96_u{n}.cce",
            folder / "camodel", s["timing"]["vf_total_cycles"], source_cycles=cmp["vfsim_source_cycles"],
            global_cycles=ab["baseline"]["vf_end_cycle"] if ab else cmp["vfsim_source_cycles"],
            local_cycles=ab["local"]["vf_end_cycle"] if ab else None,
            source_model=folder / "vfsim_source_global" if ab else None,
            global_model=replay / "ub_comparison/global" if ab else folder / "vfsim_source_global",
            local_model=replay / "ub_comparison/local" if ab else None,
            replay=replay / "source/compiled_pc_replay.cce" if ab else None,
            spill=(s["spill_store_count"], s["spill_load_count"]), membars=s["membar_count"],
            note="U12 含 32 对标量栈访存，标量时序未建模；全局/局部使用含向量 spill 的 PC 回放。" if ab else "未发现编译器向量 spill，采用源码预测。",
            evidence=(folder / "summary.json",) + ((replay / "ub_comparison/summary.json",) if ab else ())))
    cases.append({"id":"gelu_poly", "title":"GeLU Poly", "subtitle":"向量寄存器溢出", "shape":"6144 元素，FP32，96 次向量迭代",
                  "default":"u12", "headline":"U12 的 320 条动态 mem_bar 将成本放大；局部依赖预测从 6016 降到 1656 cycle。",
                  "notes":["已测档位中 U6 的 CAModel 最快，U12 首次确认 spill；未测试 U9/U10/U11。",
                           "U16 编译栈超过 6144B 上限，没有有效实测周期。"], "variants":variants})

    for case_id, title, base, numbers, shape, headline in (
        ("swiglu_grad", "SwiGLU Grad", EXPERIMENTS / "swiglu_grad_fp32_i96/unroll_scan_20260914", (1,2,4,6,8),
         "每个 tensor 6144 元素，FP32，双输出", "U6 开始出现向量 spill；U8 的局部依赖预测比全局同步快 7.21x。"),
        ("adam_apply_one", "AdamApplyOne", EXPERIMENTS / "adam_apply_one_fused_i48_unroll", (1,2,4,6,8),
         "3072 元素，FP32，48 次迭代，三输出", "U6 开始 spill；U8 保留额外访存后，局部依赖为 509 cycle。")):
        variants=[]
        for n in numbers:
            folder=base/f"u{n}"
            s=read(folder/"summary.json")
            source=next((folder/"source").glob("*.cce"))
            replay=folder/"compiled_replay"
            ab=read(replay/"summary.json") if replay.exists() else None
            cam=s["timing"]["vf_total_cycles"] if case_id=="swiglu_grad" else s["vf_cycles"]
            src=s["vfsim_source_cycles"] if case_id=="swiglu_grad" else s["vfsim_cycles"]
            stores=s["spill_store_count"] if case_id=="swiglu_grad" else s["spill_stores_by_address"]
            loads=s["spill_load_count"] if case_id=="swiglu_grad" else s["spill_loads_by_address"]
            note="采用编译后 PC 回放，保留所有向量 spill；标量指令时序未建模。" if ab else "源码级预测；未确认向量 spill。"
            if case_id=="adam_apply_one":note+=" 本轮 VSQRT 及相关参数使用 fallback。"
            if case_id=="swiglu_grad" and n==8:note+=" 本轮 VMOV 使用默认参数，另有 61 对标量栈访存。"
            variants.append(make_variant(case_id,f"u{n}",f"Unroll {n}",source,folder/"camodel",cam,
                source_cycles=src,global_cycles=ab["baseline"]["vf_end_cycle"] if ab else src,
                local_cycles=ab["local"]["vf_end_cycle"] if ab else None,
                global_model=replay/"vfsim/global" if ab else folder/("vfsim/source" if case_id=="swiglu_grad" else "vfsim"),
                local_model=replay/"vfsim/local" if ab else None,
                source_model=folder/("vfsim/source" if case_id=="swiglu_grad" else "vfsim") if ab else None,
                replay=replay/"source/compiled_pc_replay.cce" if ab else None,
                spill=(stores,loads),membars=s["membar_count"],note=note,
                evidence=(folder/"summary.json",) + ((replay/"summary.json",replay/"audit.json") if ab else ())))
        cases.append({"id":case_id,"title":title,"subtitle":"向量寄存器溢出","shape":shape,"default":"u8","headline":headline,
                      "notes":["U1/U2/U4 没有向量 spill；U6/U8 全局与局部采用相同编译后指令流。",
                               "U12/U16 编译失败，没有有效 CAModel 时间；未列入成功档位对比。"],"variants":variants})

    grad=EXPERIMENTS/"GeLU_grad/experiments"
    collected=ROOT/"cce_code/gelu_grad/log_reconstructed_cases"
    results={r["file"]:r for r in read(collected/"results.json")["cases"]}
    variants=[]
    for n in (1,2,4,8):
        filename=f"u{n}_no_explicit_spill.cce"
        r=results[filename]
        archived=grad/f"explicit_vmov_validation/u{n}"
        align=read(archived/"alignment.json")
        variants.append(make_variant("gelu_grad",f"u{n}",f"单循环 Unroll {n}",collected/filename,
            archived/"logs",r["camodel"],source_cycles=r["vfsim"],
            global_cycles=r["vfsim"] if n!=8 else None,
            global_model=collected/f"predictions/u{n}_no_explicit_spill" if n!=8 else None,
            source_model=collected/f"predictions/u{n}_no_explicit_spill" if n==8 else None,
            spill=(40,40) if n==8 else (0,0),predicate_spill=(24,24) if n==8 else (0,0),
            membars=128 if n==8 else 0,
            note="源码预测不含编译器插入的 vector/predicate spill，不能用于原二进制的精度评估。" if n==8 else "循环体与原日志一致；编译器额外插入两条循环外自拷贝，未计入预测。",
            evidence=(archived/"alignment.json",archived/"validation.json")))
    explicit=grad/"u8_explicit_spill"
    abdir=explicit/"vfsim_multisoc_membar_32b"
    ab=read(abdir/"summary.json")
    variants.append(make_variant("gelu_grad","u8_explicit","U8 · 显式 vector + predicate spill",
        collected/"u8_explicit_spill.cce",explicit/"logs",2541,global_cycles=2421,local_cycles=862,
        global_model=abdir/"global",local_model=abdir/"local",
        prediction=abdir/"prediction_input.cce",spill=(40,40),predicate_spill=(24,24),membars=128,
        note="模型输入等价声明 spill UB 参数，所有 320 次地址精确解析，512 对 RAW/WAR 审计通过。编译器另生成两条自拷贝；机器编码非完全一致。",
        evidence=(explicit/"summary.json",abdir/"summary.json",abdir/"dependency_audit.json")))
    cases.append({"id":"gelu_grad","title":"GeLU Grad","subtitle":"向量与谓词双重溢出","shape":"4096 元素，FP32，64 次向量迭代",
                  "default":"u8_explicit","headline":"U8 同时出现 vector/predicate spill；显式还原后全局精度为 95.28%。",
                  "notes":["未显式加入 spill 的 U8 源码预测仅 883 cycle，实测为 2553，不能把差距当作模型内同步收益。",
                           "显式 spill 源码实测 2541，全局预测 2421；32B 局部依赖为 862（假设硬件能力）。"],"variants":variants})
    three=grad/"three_stage_log_aligned"
    three_v=make_variant("gelu_grad_three_stage","three_stage_post","三段循环 · POST_UPDATE · 每段 U1",
        collected/"three_stage_post_update.cce",three/"logs",854,global_cycles=932,
        global_model=collected/"predictions/three_stage_post_update",membars=2,
        note="每段 64 次循环，指令体分别为 10/11/8 条；两条 VST_VLD。按原日志复用寄存器，无 vector/predicate/scalar spill。",
        evidence=(three/"summary.json",three/"validation.json"))
    # Separate copies make this case portable without another case directory.
    comparison=[]
    for n in (1,4):
        r=results[f"u{n}_no_explicit_spill.cce"]
        comparison.append(make_variant("gelu_grad_three_stage",f"single_u{n}",f"单循环 U{n} 参考",
            collected/f"u{n}_no_explicit_spill.cce",grad/f"explicit_vmov_validation/u{n}/logs",r["camodel"],
            global_cycles=r["vfsim"],global_model=collected/f"predictions/u{n}_no_explicit_spill",
            note="原始单循环参考，显式 VMOV；未发生 spill。"))
    cases.append({"id":"gelu_grad_three_stage","title":"GeLU Grad 三段循环","subtitle":"循环切分与 POST_UPDATE",
                  "shape":"4096 元素，FP32；显式 UB 布局 96KiB","default":"three_stage_post",
                  "headline":"三段切分保留同步仍比单循环 U1 快 1.17x；相对 U4 快 1.04x。",
                  "notes":["CAModel 比较的是实际运行代码：单循环 U1 996、U4 891、三段 POST_UPDATE 854。",
                           "每段实际 U1，不把历史 pragma U8 误写为硬件展开 8 次。额外 scratch 增加 UB 占用。"],
                  "variants":comparison+[three_v], "overlay":True})

    for case in cases:
        (DEST/case["id"] / "summary.json").write_text(json.dumps(case,ensure_ascii=False,indent=2))
    data={"generated":"2026-10-08", "soc":"A5 / DV100 / dav-3510", "window":10,
          "ipc_method":"向量计算指令完成事件，10-cycle trailing window，各曲线以首条计算完成为 cycle 0；不含 LSU、Membar、PSET、scalar。",
          "cases":cases}
    (DEST/"data.json").write_text(json.dumps(data,ensure_ascii=False,separators=(",",":")))
    (DEST/"data.js").write_text("window.MEMBAR_REPORT="+json.dumps(data,ensure_ascii=False,separators=(",",":"))+";\n")
    (DEST/"manifest.json").write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2))
    with (DEST/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.writer(stream)
        writer.writerow(["case","variant","CAModel","源码预测","全局预测","局部预测","全局精度%","模型内加速倍数","vector写","vector读","predicate写","predicate读","动态Membar"])
        for case in cases:
            for v in case["variants"]:
                writer.writerow([case["title"],v["label"],v["camodel"],v["source_cycles"],v["global_cycles"],v["local_cycles"],v["accuracy"],v["speedup"],v["vector_store"],v["vector_load"],v["predicate_store"],v["predicate_load"],v["membars"]])
    print(json.dumps({"cases":len(cases),"variants":sum(len(c["variants"]) for c in cases),"verified_copies":len(MANIFEST)},ensure_ascii=False))


if __name__ == "__main__":
    collect()
    from prepare_hardware_equivalent import bundle
    bundle(DEST)
