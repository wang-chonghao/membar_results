"""Copy only the HTML to an empty folder and check it in an offline browser."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=Path(__file__).resolve().parents[1] / "spill_candidate_alignment_standalone.html")
    parser.add_argument("--out", type=Path, default=Path("/tmp/membar-single-report-browser"))
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="membar-html-only-") as temporary:
        html = Path(temporary) / "report.html"
        shutil.copyfile(args.html, html)
        errors, dependencies = [], []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                page = browser.new_page(viewport={"width": 1920, "height": 1100}, offline=True, accept_downloads=True)
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda req: dependencies.append(req.url) if req.url != html.as_uri() and not req.url.startswith(("data:", "blob:")) else None)
                page.goto(html.as_uri(), timeout=120_000)
                page.wait_for_function("window.EMBEDDED_REPORT?.ready && window.SPILL_CHART?.()", timeout=120_000)
                font = Path("/mnt/c/Windows/Fonts/msyh.ttc")
                if font.exists():
                    encoded = base64.b64encode(font.read_bytes()).decode()
                    page.add_style_tag(content=f'@font-face {{font-family: "Report Test CJK"; src: url(data:font/ttf;base64,{encoded});}} body {{font-family: "Report Test CJK",sans-serif;}}')
                    page.evaluate("document.fonts.ready")
                    page.evaluate("Chart.defaults.font.family = 'Report Test CJK'")
                count = page.evaluate("Object.keys(EMBEDDED_REPORT.files).length")
                assert count > 7000, count
                assert page.locator("#results-body tr").count() == 92
                for arch in ("A6", "A5", "A6"):
                    page.select_option("#arch-filter", arch)
                    assert arch in page.locator("#architecture-note").inner_text()
                    assert page.evaluate("SPILL_CHART().data.datasets.length") == 4
                    expected = page.evaluate("a => MULTISOC_REPORT.rows['swiglu-u4'].arches[a].original.global_cycles", arch)
                    assert page.locator('tr[data-id="swiglu-u4"] [data-metric="global_cycles"]').inner_text() == str(expected)
                    if page.evaluate("Boolean(window.SPILL_PERFORMANCE_SUMMARY)"):
                        assert page.locator("#spill-performance-body tr").count() == (4 if arch == "A6" else 8)
                        if arch == "A6":
                            expected_software = page.evaluate("SPILL_PERFORMANCE_SUMMARY.rows['swiglu-u4'].A6.software_cycles")
                            assert page.locator('tr[data-id="swiglu-u4"] [data-metric="software_cycles"]').inner_text() == str(expected_software)
                        approximations = page.evaluate("Object.values(SPILL_PERFORMANCE_SUMMARY.rows).filter(r => r.A6.scalar_spill_excluded).map(r => r.id)")
                        for key in approximations:
                            assert page.locator(f'tr[data-id="{key}"] .scalar-timing-note').count() == 1
                page.select_option("#version-scope", "all")
                assert page.locator("#results-body tr").count() == 153
                page.select_option("#version-scope", "main")
                page.select_option("#case-filter", "swiglu")
                optimized = page.locator('tr[data-id="swiglu-u4"] .software-cce-link')
                if optimized.count():
                    optimized_path = optimized.first.get_attribute("href")
                    optimized.first.click()
                    page.wait_for_function("document.querySelector('#embedded-body pre') !== null")
                    assert optimized_path in page.locator("#embedded-title").inner_text()
                    assert "Selected A6 software schedule" in page.locator("#embedded-body pre").inner_text()
                    page.locator("#embedded-close").click()
                source = page.locator('#results-body tr[data-id="swiglu-u4"] :is(th,td):first-child a').first
                path = source.get_attribute("href")
                source.click()
                page.wait_for_function("document.querySelector('#embedded-body pre') !== null")
                assert "v" in page.locator("#embedded-body pre").inner_text()
                assert path in page.locator("#embedded-title").inner_text()
                with page.expect_download() as download_info:
                    page.locator("#embedded-download").click()
                saved = download_info.value.path()
                expected = page.evaluate("p => EMBEDDED_REPORT.files[p].sha256", path)
                assert hashlib.sha256(Path(saved).read_bytes()).hexdigest() == expected
                page.locator("#embedded-close").click()
                if page.evaluate("Boolean(window.REPLAY_VALIDATION)"):
                    record = page.evaluate("REPLAY_VALIDATION.rows['swiglu-u4']")
                    assert record['status'] in ('strict_pass', 'equivalent_pass'), record
                    replay = page.locator(f'#results-body tr[data-id="swiglu-u4"] a[href="{record["source_path"]}"]')
                    replay.click()
                    page.wait_for_function("document.querySelector('#embedded-body pre') !== null")
                    with page.expect_download() as replay_download:
                        page.locator("#embedded-download").click()
                    assert hashlib.sha256(Path(replay_download.value.path()).read_bytes()).hexdigest() == record['source_sha256']
                    page.locator("#embedded-close").click()
                page.locator('.detail-button[data-id="swiglu-u4"]').click()
                page.locator('#detail-body a[href^="multisoc_refresh_"]').first.click()
                page.wait_for_function("document.querySelector('#embedded-body pre')?.textContent.includes('A6')")
                page.locator("#embedded-close").click()
                page.locator("#close-detail").click()
                image = page.evaluate("Object.keys(EMBEDDED_REPORT.files).find(p => p.startsWith('figure/DAG/') && p.endsWith('.png')) || Object.keys(EMBEDDED_REPORT.files).find(p => p.startsWith('figure/') && p.endsWith('.png'))")
                page.evaluate("p => EMBEDDED_REPORT.open(p)", image)
                page.wait_for_function("document.querySelector('#embedded-body img')?.naturalWidth > 0")
                page.locator("#embedded-body img").screenshot(path=str(args.out / "embedded_image.png"))
                page.locator("#embedded-close").click()
                page.locator('header nav a[href="a6_softmax_fusion_results_summary.html"]').click()
                page.wait_for_function("document.querySelector('#embedded-body iframe') !== null")
                frame = page.frame_locator("#embedded-body iframe")
                assert "Softmax" in frame.locator("h1").inner_text()
                assert frame.locator("img").count() > 0
                for image_node in frame.locator("img").all():
                    image_node.scroll_into_view_if_needed()
                    image_node.evaluate("img => img.decode()")
                    assert image_node.evaluate("img => img.complete && img.naturalWidth > 0")
                frame.locator('a[href$=".cce"]').first.click()
                page.wait_for_function("document.querySelector('#embedded-body pre') !== null")
                page.locator("#embedded-close").click()
                dag_link = page.locator('header nav a[href="a6_spill_dag_comparison_20261027.html"]')
                if dag_link.count():
                    dag_link.click()
                    page.wait_for_function("document.querySelector('#embedded-body iframe') !== null")
                    dag_frame = page.frame_locator("#embedded-body iframe")
                    assert "A6 Spill DAG" in dag_frame.locator("h1").inner_text()
                    assert dag_frame.locator("img").count() == 10
                    for diagram in dag_frame.locator("img").all():
                        diagram.scroll_into_view_if_needed()
                        diagram.evaluate("img => img.decode()")
                        assert diagram.evaluate("img => img.complete && img.naturalWidth > 0")
                    dag_frame.locator('a[href$="a6_logit_u3_local_instruction_order_20261027.png"]').click()
                    page.wait_for_function("document.querySelector('#embedded-body img')?.naturalWidth > 0")
                    page.locator("#embedded-body img").screenshot(path=str(args.out / "logit_u3_dag.png"))
                    page.locator("#embedded-close").click()
                page.locator('header nav a[href="#embedded-files"]').click()
                page.fill("#embedded-search", "figure/cycles/")
                assert page.locator(".embedded-file-table tr").count() > 0
                page.locator("#embedded-close").click()
                page.select_option("#case-filter", "all")
                page.select_option("#chart-case", "swiglu")
                page.locator("#cycles-chart").screenshot(path=str(args.out / "standalone_chart.png"))
                page.evaluate("window.scrollTo(0,0)")
                page.screenshot(path=str(args.out / "standalone_desktop.png"))
                assert not errors, errors
                assert not dependencies, dependencies
                result = dict(passed=True, embedded_files=count, rows=153, external_dependencies=dependencies, errors=errors, html_bytes=html.stat().st_size)
                (args.out / "verification.json").write_text(json.dumps(result, indent=2))
                print(json.dumps(result), flush=True)
            finally:
                browser.close()


if __name__ == "__main__":
    main()
