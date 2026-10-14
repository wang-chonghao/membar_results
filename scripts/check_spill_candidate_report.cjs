"use strict";

const {chromium} = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {pathToFileURL} = require("node:url");

const root = path.resolve(process.argv[2] || path.join(__dirname, ".."));
const output = path.resolve(process.argv[3] || "/tmp/spill-candidate-report-check");

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const data = JSON.parse(fs.readFileSync(path.join(root, "metadata/spill_candidate_alignment.json"), "utf8"));
  const errors = [], external = [];
  const browser = await chromium.launch({headless: true, args: ["--no-sandbox"]});
  try {
    const context = await browser.newContext({viewport: {width: 1600, height: 1100}, offline: true});
    const page = await context.newPage();
    page.on("pageerror", error => errors.push(String(error)));
    page.on("request", request => {
      if (!request.url().startsWith("file:") && !request.url().startsWith("data:") && !request.url().startsWith("blob:")) external.push(request.url());
    });
    await page.goto(pathToFileURL(path.join(root, "spill_candidate_alignment_summary.html")).href);
    await page.waitForFunction(() => window.SPILL_CHART?.());
    assert.equal(await page.locator("#results-body tr").count(), data.rows.length);
    assert.equal(await page.locator('option[value="failed"]').count(), 0);
    assert.equal(data.rows.some(row => row.status === "vf_stack_exceeded"), false);
    const headers = await page.locator("thead th").allTextContents();
    assert.ok(headers.some(header => header.includes("VfSim 软件优化全局")));
    assert.equal(headers.findIndex(header=>header.trim().startsWith("U ")),headers.findIndex(header=>header.includes("循环次数"))+1);
    assert.equal(await page.locator('tr[data-id="swiglu-u4"] td.num[title]').innerText(),"16");
    assert.equal(await page.locator('tr[data-id="history-gelu_poly-u12"] td.num[title]').innerText(),"96");
    assert.match(await page.locator('tr[data-id="history-mhc_sinkhorn-m128_element_major_u2"] td.num[title]').innerText(),/批次 2；迭代 19/);
    assert.match(await page.locator('tr[data-id="swiglu-u4"] .status').innerText(),/Vector \+ Predicate spill/);
    assert.match(await page.locator('tr[data-id="swiglu-u4"] .status').innerText(),/Vector 写 12 \/ 读 12/);
    assert.match(await page.locator('tr[data-id="swiglu-u4"] .status').innerText(),/Predicate 写 20 \/ 读 20/);
    assert.match(await page.locator('tr[data-id="focal-u2"] .status').innerText(),/Predicate spill/);
    assert.match(await page.locator('tr[data-id="groupnorm-u8"] .status').innerText(),/Predicate 写 6 \/ 读 34/);
    assert.equal(headers.findIndex(header => header.includes("Membar / 软件性能")), headers.findIndex(header => header.includes("优化后精度")) + 1);
    assert.equal(headers.findIndex(header => header.includes("软件策略")), headers.findIndex(header => header.includes("Membar / 软件性能")) + 1);
    assert.match(await page.locator('tr[data-id="swiglu-u4"]').innerText(), /86\.55%/);
    assert.match(await page.locator('tr[data-id="groupnorm-u8"]').innerText(), /101\.72%/);
    const retained = await page.locator('tr[data-id="swiglu-u1"]').innerText();
    assert.equal((retained.match(/550/g) || []).length, 2);
    assert.equal((retained.match(/98\.18%/g) || []).length, 2);
    await page.screenshot({path: path.join(output, "desktop.png"), fullPage: true});
    await page.selectOption("#origin-filter", "supplemental_search");
    assert.equal(await page.locator("#results-body tr").count(), 10);
    assert.equal(await page.locator('tr[data-id="round2-logsigmoid_grad-u8"] td.num[title]').innerText(), "48");
    assert.match(await page.locator('tr[data-id="round2-logsigmoid_grad-u8"]').innerText(), /3781/);
    if (data.round3_import) {
      await page.selectOption("#origin-filter", "round3_search");
      assert.equal(await page.locator("#results-body tr").count(), data.round3_import.selected_versions);
      assert.equal(await page.locator('tr[data-id="round3-groupnorm_swish_dx-u4"] td.num[title]').innerText(), "48");
      assert.match(await page.locator('tr[data-id="round3-groupnorm_swish_dx-u4"]').innerText(), /12638/);
      assert.match(await page.locator('tr[data-id="round3-groupnorm_swish_dx-u4"]').innerText(), /1430/);
      assert.match(await page.locator('tr[data-id="round3-rope_grad-u4"] .status').innerText(), /Scalar spill/);
      assert.match(await page.locator('tr[data-id="round3-rope_grad-u4"] .status').innerText(), /Scalar 写 84 \/ 读 84/);
      await page.locator('.detail-button[data-id="round3-causal_conv-u1"]').click();
      assert.match(await page.locator("#detail-body").innerText(), /手工 JSON/);
      assert.match(await page.locator("#detail-body").innerText(), /全零 mask/);
      assert.match(await page.locator('tr[data-id="round3-causal_conv-u2"]').innerText(), /4361/);
      assert.match(await page.locator('tr[data-id="round3-causal_conv-u2"]').innerText(), /1095/);
      await page.locator("#close-detail").click();
      await page.selectOption("#case-filter", "softplus");
      await page.locator('.detail-button[data-id="round3-softplus-u1"]').click();
      assert.match(await page.locator("#detail-body").innerText(), /1869/);
      assert.match(await page.locator("#detail-body").innerText(), /515/);
      await page.locator("#close-detail").click();
      await page.selectOption("#case-filter", "all");
      await page.screenshot({path: path.join(output, "round3.png"), fullPage: true});
    }
    await page.selectOption("#origin-filter","fixed_unroll_search");
    assert.equal(await page.locator("#results-body tr").count(),33);
    for (const [filter, count] of [["spill", 23], ["no-spill", 10], ["unsupported", 3]]) {
      await page.selectOption("#status-filter", filter);
      assert.equal(await page.locator("#results-body tr").count(), count);
    }
    await page.selectOption("#status-filter", "all");
    await page.selectOption("#origin-filter","historical_membar");
    assert.equal(await page.locator("#results-body tr").count(),24);
    const geluReplay = data.rows.find(row => row.id === "history-gelu_grad-u8");
    if (geluReplay.compiled_replay_update) {
      assert.equal(geluReplay.global_cycles, 2421);
      assert.equal(geluReplay.local_cycles, 862);
      assert.match(await page.locator('tr[data-id="history-gelu_grad-u8"]').innerText(), /94\.83%/);
      await page.locator('.detail-button[data-id="history-gelu_grad-u8"]').click();
      assert.match(await page.locator("#detail-body").innerText(), /完整 PC 流/);
      await page.locator("#close-detail").click();
    }
    await page.selectOption("#case-filter","flash_attention_grad_front");
    assert.equal(await page.locator("#results-body tr").count(),2);
    await page.locator('.detail-button[data-id="history-flash_attention_grad_front-u2_concentrated_spill"]').click();
    assert.match(await page.locator("#detail-body").innerText(),/201 cycles/);
    assert.match(await page.locator("#detail-body").innerText(),/234 cycles/);
    assert.match(await page.locator("#detail-body").innerText(),/208 cycles 不包含完整 compiler spill/);
    await page.locator("#close-detail").click();
    await page.screenshot({path:path.join(output,"historical_flash_attention.png"),fullPage:true});
    await page.selectOption("#case-filter","mhc_sinkhorn");
    await page.fill("#search","128");
    assert.equal(await page.locator("#results-body tr").count(),2);
    assert.equal(await page.locator("#chart-case").inputValue(),"mhc_sinkhorn");
    assert.equal(await page.locator("#chart-group").inputValue(),"mhc-128-calibrated");
    await page.screenshot({path:path.join(output,"historical_mhc.png"),fullPage:true});
    await page.fill("#search","");
    await page.selectOption("#case-filter","all");
    await page.selectOption("#origin-filter","all");
    await page.fill("#search", "not-a-case");
    assert.equal(await page.locator("#results-body tr").count(), 0);
    assert.equal(await page.locator("#empty-state").isVisible(), true);
    await page.fill("#search", "");
    await page.selectOption("#case-filter", "focal");
    assert.equal(await page.locator("#results-body tr").count(), 4);
    await page.locator('[data-sort="global_cycles"]').click();
    assert.deepEqual(await page.locator("#results-body tr").evaluateAll(rows => rows.map(row => row.dataset.id)),
      ["focal-u1", "focal-u2", "focal-u4", "focal-u5"]);
    await page.locator('.detail-button[data-id="focal-u4"]').click();
    await page.waitForSelector("#detail-dialog[open]");
    assert.match(await page.locator("#detail-body").innerText(), /355 cycles/);
    assert.match(await page.locator("#detail-body").innerText(), /356 cycles/);
    await page.keyboard.press("Escape");
    await page.selectOption("#case-filter", "all");
    await page.selectOption("#status-filter", "unsupported");
    await page.locator('.detail-button[data-id="adam_decay-u7"]').click();
    assert.match(await page.locator("#detail-body").innerText(), /RV_SSTI\/SLDI/);
    assert.match(await page.locator("#detail-body").innerText(), /249 cycles/);
    await page.locator("#close-detail").click();
    await page.selectOption("#status-filter", "all");

    let links = 0;
    for (const row of data.rows) {
      const primary=page.locator(`tr[data-id="${row.id}"] .cce-link`);
      // Check filtered-out rows from the preserved DOM-independent metadata below.
      if(await primary.count())assert.equal(await primary.getAttribute("href"),row.source_cce_path);
      assert.equal(fs.existsSync(path.join(root,row.source_cce_path)),true);
      const code=fs.readFileSync(path.join(root,row.source_cce_path),"utf8").replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g,"");
      assert.doesNotMatch(code,/\b(?:AscendC|MicroAPI|Reg)::|\bRegTensor\s*</);
      if(row.pure_cce_refresh){
        assert.equal(row.source_cce_path,`spill_candidate_alignment/pure_cce/${row.id}/cce/kernel.cce`);
        for(const key of ["camodel","global_cycles","local_cycles"])assert.equal(row[key],row.pure_cce_refresh[key]);
      }
      for (const entries of Object.values(row.files)) {
        for (const file of entries) {
          assert.equal(fs.existsSync(path.join(root, file.path)), true, file.path);
          assert.equal(path.resolve(root, file.path).startsWith(root + path.sep), true);
          links++;
        }
      }
      assert.equal(fs.existsSync(path.join(root, row.evidence_path)), true);
    }
    await page.selectOption("#case-filter","swiglu");
    const firstLink=page.locator('tr[data-id="swiglu-u1"] .cce-link');
    const popup=page.waitForEvent("popup");
    await firstLink.click();
    const opened=await popup;
    await opened.waitForLoadState("domcontentloaded");
    assert.ok(opened.url().endsWith("spill_candidate_alignment/swiglu/u1/original/cce/kernel.cce"));
    assert.match(await opened.locator("body").innerText(),/candidate_vf/);
    await opened.close();
    await page.selectOption("#case-filter","all");
    let plotGroups=0;
    for (const operator of data.case_order) {
      await page.selectOption("#chart-case", operator);
      const groups=await page.locator("#chart-group option").evaluateAll(options=>options.map(option=>option.value));
      for(const group of groups) {
      plotGroups++;
      await page.selectOption("#chart-group",group);
      const chart = await page.evaluate(() => {
        const graph = window.SPILL_CHART(), canvas = document.querySelector("canvas");
        const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
        let colored = 0;
        for (let i = 0; i < pixels.length; i += 4) if (pixels[i + 3] && Math.max(...pixels.slice(i, i + 3)) - Math.min(...pixels.slice(i, i + 3)) > 30) colored++;
        return {labels: graph.data.labels, values: graph.data.datasets.map(series => series.data), colored};
      });
      const rows = data.rows.filter(row => row.case === operator&&row.comparison_group===group).sort((a, b) => a.variant_order - b.variant_order);
      assert.deepEqual(chart.labels, rows.map(row => row.chart_label));
      assert.deepEqual(chart.values, ["camodel", "global_cycles", "local_cycles", "software_cycles"].map(key => rows.map(row => row[key])));
      assert.ok(chart.colored > 100, "Nonblank chart: " + operator);
      }
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
    const result = {rows: data.rows.length, cases: data.case_order.length, verified_links: links, charts: plotGroups, offline: true,
      missing_values_preserved: true, filters: true, sorting: true, detail_dialog: true,
      viewport_widths: [1600], page_errors: errors, external_requests: external};
    fs.writeFileSync(path.join(output, "verification.json"), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
