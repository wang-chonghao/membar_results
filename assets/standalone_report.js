"use strict";

(async () => {
  const archive = JSON.parse(document.getElementById("embedded-index").textContent);
  const files = archive.files;
  const names = Object.keys(files).sort();
  const previewLimit = 1024 * 1024;
  const textCache = new Map();
  let activePath = null, request = 0, imageUrl = null, childFrame = null, page = 0;
  const pageSize = 100;

  const status = document.createElement("div");
  status.id = "embedded-loading";
  status.textContent = "正在读取内嵌报表…";
  document.body.prepend(status);
  const dialog = document.createElement("dialog");
  dialog.id = "embedded-dialog";
  dialog.setAttribute("aria-label", "内嵌附件");
  dialog.innerHTML = `<div class="embedded-toolbar"><button id="embedded-back" title="返回内嵌文件目录">文件目录</button><h2 id="embedded-title"></h2><input id="embedded-search" type="search" placeholder="文件路径" aria-label="搜索内嵌文件"><button id="embedded-download">下载文件</button><button id="embedded-close" title="关闭附件" aria-label="关闭附件">&times;</button></div><div id="embedded-body"></div><div id="embedded-footer"><span id="embedded-status"></span><span><button id="embedded-prev">上一页</button> <button id="embedded-next">下一页</button></span></div>`;
  document.body.append(dialog);
  const body = dialog.querySelector("#embedded-body");
  const title = dialog.querySelector("#embedded-title");
  const footer = dialog.querySelector("#embedded-status");
  const search = dialog.querySelector("#embedded-search");
  const download = dialog.querySelector("#embedded-download");
  const prev = dialog.querySelector("#embedded-prev"), next = dialog.querySelector("#embedded-next");

  function resolve(reference, owner = archive.entry) {
    if (!reference || reference.startsWith("#")) return null;
    const url = new URL(reference, new URL(owner, "https://embedded.invalid/"));
    if (url.origin !== "https://embedded.invalid") return null;
    return decodeURIComponent(url.pathname.slice(1));
  }

  async function bytes(path) {
    const info = files[path];
    if (!info) throw new Error(`未内嵌文件：${path}`);
    const encoded = document.getElementById(info.blob).textContent.trim();
    const binary = atob(encoded);
    const packed = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) packed[i] = binary.charCodeAt(i);
    const decoded = new Response(new Blob([packed]).stream().pipeThrough(new DecompressionStream("gzip")));
    const result = new Uint8Array(await decoded.arrayBuffer());
    if (result.length !== info.size) throw new Error(`附件长度校验失败：${path}`);
    return result;
  }

  async function text(path) {
    if (textCache.has(path)) return textCache.get(path);
    const value = new TextDecoder().decode(await bytes(path));
    if (files[path].size < 2 * previewLimit) textCache.set(path, value);
    return value;
  }

  const size = n => n >= 2**20 ? `${(n / 2**20).toFixed(2)} MiB` : n >= 1024 ? `${(n / 1024).toFixed(1)} KiB` : `${n} B`;
  function clearPreview() {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = null;
    childFrame = null;
    body.replaceChildren();
  }
  function showDialog() { if (!dialog.open) dialog.showModal(); }
  function list() {
    ++request;
    activePath = null;
    clearPreview();
    title.textContent = "内嵌文件";
    search.hidden = false;
    download.hidden = true;
    const matches = names.filter(name => name.toLowerCase().includes(search.value.trim().toLowerCase()));
    page = Math.max(0, Math.min(page, Math.ceil(matches.length / pageSize) - 1));
    const table = document.createElement("table");
    table.className = "embedded-file-table";
    for (const name of matches.slice(page * pageSize, (page + 1) * pageSize)) {
      const row = table.insertRow(), cell = row.insertCell(), link = document.createElement("a");
      link.href = name;
      link.textContent = name;
      cell.append(link);
      row.insertCell().textContent = size(files[name].size);
    }
    body.append(table);
    footer.textContent = `${matches.length} / ${names.length} 个文件 · 第 ${page + 1} / ${Math.max(1, Math.ceil(matches.length / pageSize))} 页`;
    prev.hidden = next.hidden = false;
    prev.disabled = page === 0;
    next.disabled = (page + 1) * pageSize >= matches.length;
    showDialog();
  }

  async function htmlPreview(path) {
    const doc = new DOMParser().parseFromString(await text(path), "text/html");
    for (const node of doc.querySelectorAll('link[rel="stylesheet"]')) {
      const style = doc.createElement("style");
      style.textContent = await text(resolve(node.getAttribute("href"), path));
      node.replaceWith(style);
    }
    const scripts = [];
    for (const node of doc.querySelectorAll("script[src]")) {
      const script = doc.createElement("script");
      script.textContent = await text(resolve(node.getAttribute("src"), path));
      scripts.push(script);
      node.remove();
    }
    for (const node of doc.querySelectorAll("img[src],source[src],video[poster]")) {
      const key = node.hasAttribute("src") ? "src" : "poster";
      const resolved = resolve(node.getAttribute(key), path);
      if (!resolved) continue;
      const data = await bytes(resolved);
      const encoded = await new Promise(done => {
        const reader = new FileReader();
        reader.onload = () => done(reader.result);
        reader.readAsDataURL(new Blob([data], {type: files[resolved].mime}));
      });
      node.setAttribute(key, encoded);
    }
    const base = doc.createElement("base");
    base.href = `https://embedded.invalid/${path}`;
    doc.head.prepend(base);
    const bridge = doc.createElement("script");
    bridge.textContent = `document.addEventListener('click', event => {const a=event.target.closest('a[href]');if(!a)return;const raw=a.getAttribute('href');if(raw.startsWith('#'))return;const u=new URL(raw,document.baseURI);if(u.origin==='https://embedded.invalid'){event.preventDefault();parent.postMessage({embeddedFile:decodeURIComponent(u.pathname.slice(1))},'*');}});`;
    doc.body.append(bridge, ...scripts);
    return "<!doctype html>\n" + doc.documentElement.outerHTML;
  }

  async function open(path) {
    if (path === archive.entry) { dialog.close(); return; }
    const token = ++request;
    activePath = path;
    clearPreview();
    title.textContent = path;
    search.hidden = true;
    download.hidden = !files[path];
    prev.hidden = next.hidden = true;
    footer.textContent = files[path] ? `${size(files[path].size)} · SHA-256 ${files[path].sha256}` : "";
    body.textContent = "正在解压附件…";
    showDialog();
    try {
      if (!files[path] && archive.bundle_mode === "lean" && archive.omitted?.[path]) {
        body.textContent = `精简版未收录：${archive.omitted[path]}。此文件属于详细复现证据，不影响表格、图表或源码浏览。路径：${path}`;
        return;
      }
      if (!files[path]) throw new Error(`未内嵌文件：${path}`);
      if (files[path].mime.startsWith("text/html")) {
        const source = await htmlPreview(path);
        if (token !== request) return;
        clearPreview();
        childFrame = document.createElement("iframe");
        childFrame.title = path;
        childFrame.setAttribute("sandbox", "allow-scripts");
        childFrame.srcdoc = source;
        body.append(childFrame);
        return;
      }
      const data = await bytes(path);
      if (token !== request) return;
      clearPreview();
      if (files[path].mime.startsWith("image/")) {
        imageUrl = URL.createObjectURL(new Blob([data], {type: files[path].mime}));
        const img = document.createElement("img");
        img.alt = path;
        img.src = imageUrl;
        body.append(img);
      } else if (files[path].mime.startsWith("text/") || /\.(json|jsonl|yaml|yml)$/.test(path)) {
        if (data.length > previewLimit) {
          const note = document.createElement("p");
          note.className = "embedded-warning";
          note.textContent = "预览前 1 MiB；完整内容已内嵌，可下载原始文件。";
          body.append(note);
        }
        const pre = document.createElement("pre");
        pre.textContent = new TextDecoder().decode(data.subarray(0, previewLimit));
        body.append(pre);
      } else {
        body.textContent = "二进制附件已内嵌，可下载原始文件。";
      }
    } catch (error) {
      if (token === request) body.textContent = error.message;
    }
  }

  document.addEventListener("click", event => {
    const link = event.target.closest("a[href]");
    if (!link || event.defaultPrevented) return;
    if (link.getAttribute("href") === "#embedded-files") { event.preventDefault(); list(); return; }
    const path = resolve(link.getAttribute("href"));
    if (path) { event.preventDefault(); open(path); }
  });
  window.addEventListener("message", event => {
    if (childFrame && event.source === childFrame.contentWindow && typeof event.data?.embeddedFile === "string") open(event.data.embeddedFile);
  });
  dialog.querySelector("#embedded-close").onclick = () => dialog.close();
  dialog.addEventListener("close", () => { ++request; clearPreview(); });
  dialog.querySelector("#embedded-back").onclick = list;
  search.oninput = () => { page = 0; list(); };
  prev.onclick = () => { --page; list(); };
  next.onclick = () => { ++page; list(); };
  download.onclick = async () => {
    const path = activePath;
    if (!files[path]) return;
    const url = URL.createObjectURL(new Blob([await bytes(path)], {type: files[path].mime}));
    const link = document.createElement("a");
    link.href = url;
    link.download = path.split("/").pop();
    // A detached link avoids the archive's own navigation handler.
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  };

  const nav = document.querySelector("header nav");
  if (nav) {
    const link = document.createElement("a");
    link.href = "#embedded-files";
    link.textContent = `${archive.bundle_mode === "lean" ? "精简版 · " : ""}内嵌文件（${names.length}）`;
    nav.prepend(link);
  }
  try {
    if (typeof DecompressionStream === "undefined") throw new Error("请使用支持 DecompressionStream 的新版 Chrome、Edge 或 Firefox 打开此文件。");
    for (const path of archive.scripts) {
      const script = document.createElement("script");
      script.textContent = await text(path);
      document.head.append(script);
    }
    status.remove();
    window.EMBEDDED_REPORT = {files, omitted: archive.omitted || {}, bundleMode: archive.bundle_mode || "full", readBytes: bytes, open, list, ready: true};
  } catch (error) {
    status.textContent = `单文件报表加载失败：${error.message}`;
  }
})();
