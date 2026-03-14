---
title: VLA 推荐阅读
hide:
  - date
---

# VLA 推荐阅读

> 这里整理了一些我在学习 **VLA（Vision-Language-Action）** 过程中值得反复看的中文资料

<div class="vla-read-list">
  <a class="vla-read-card" href="https://blog.csdn.net/v_JULY_v/article/details/143472442" target="_blank" rel="noopener noreferrer">
    <div class="vla-read-body">
      <div class="vla-read-glow"></div>
      <div class="vla-read-head">
        <span class="vla-read-tag">CSDN · 论文精度</span>
        <span class="vla-read-arrow">↗</span>
      </div>
      <h3>OpenPi开山之作</h3>
      <p>放在最前面读，这篇文章对Pi0的论文介绍的非常详细。</p>
      <div class="vla-read-footer">
        <strong>点击跳转</strong>
      </div>
    </div>
  </a>

  <a class="vla-read-card" href="https://blog.csdn.net/v_JULY_v/article/details/146068251" target="_blank" rel="noopener noreferrer">
    <div class="vla-read-body">
      <div class="vla-read-glow"></div>
      <div class="vla-read-head">
        <span class="vla-read-tag">CSDN · 代码详解</span>
        <span class="vla-read-arrow">↗</span>
      </div>
      <h3>OpenPi开山之作</h3>
      <p>关于代码的详细解释</p>
      <div class="vla-read-footer">
        <strong>点击跳转</strong>
      </div>
    </div>
  </a>

</div>


<style>
.vla-read-list {
  display: flex;
  flex-direction: column;
  gap: 22px;
  margin: 1.4rem 0 1.8rem;
}

.vla-read-card {
  position: relative;
  display: block;
  overflow: hidden;
  border-radius: 24px;
  text-decoration: none !important;
  color: inherit !important;
  background: linear-gradient(135deg, rgba(15,23,42,.96), rgba(30,41,59,.92));
  border: 1px solid rgba(148,163,184,.18);
  box-shadow:
    0 10px 30px rgba(2,6,23,.22),
    inset 0 1px 0 rgba(255,255,255,.06);
  transition: transform .28s ease, box-shadow .28s ease, border-color .28s ease;
}

.vla-read-card::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(120deg, transparent 20%, rgba(255,255,255,.08) 50%, transparent 80%);
  transform: translateX(-120%);
  transition: transform .8s ease;
}

.vla-read-card:hover {
  transform: translateY(-6px) scale(1.005);
  border-color: rgba(96,165,250,.36);
  box-shadow:
    0 18px 48px rgba(37,99,235,.22),
    0 0 0 1px rgba(96,165,250,.12),
    inset 0 1px 0 rgba(255,255,255,.08);
}

.vla-read-card:hover::after {
  transform: translateX(120%);
}

.vla-read-body {
  position: relative;
  padding: 24px;
  background:
    radial-gradient(circle at top right, rgba(96,165,250,.22), transparent 28%),
    radial-gradient(circle at left center, rgba(139,92,246,.18), transparent 24%);
}

.vla-read-glow {
  position: absolute;
  right: -40px;
  top: -40px;
  width: 160px;
  height: 160px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(59,130,246,.28), transparent 70%);
  pointer-events: none;
}

.vla-read-head,
.vla-read-footer {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.vla-read-head {
  margin-bottom: 12px;
}

.vla-read-tag {
  display: inline-flex;
  align-items: center;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: .78rem;
  letter-spacing: .02em;
  color: #dbeafe;
  background: rgba(59,130,246,.18);
  border: 1px solid rgba(96,165,250,.22);
  backdrop-filter: blur(8px);
}

.vla-read-arrow {
  font-size: 1.2rem;
  color: #93c5fd;
  transform: rotate(0deg);
  transition: transform .25s ease;
}

.vla-read-card:hover .vla-read-arrow {
  transform: translate(2px, -2px) scale(1.08);
}

.vla-read-card h3 {
  position: relative;
  z-index: 1;
  margin: 0 0 10px;
  font-size: 1.28rem;
  line-height: 1.45;
  color: #f8fafc;
}

.vla-read-card p {
  position: relative;
  z-index: 1;
  margin: 0 0 18px;
  line-height: 1.85;
  color: rgba(226,232,240,.88);
}

.vla-read-footer {
  padding-top: 14px;
  border-top: 1px solid rgba(148,163,184,.16);
  color: rgba(191,219,254,.88);
  font-size: .92rem;
}

.vla-read-footer strong {
  color: #ffffff;
  padding: 8px 14px;
  border-radius: 999px;
  background: linear-gradient(135deg, #2563eb, #7c3aed);
  box-shadow: 0 8px 20px rgba(59,130,246,.28);
  font-size: .9rem;
}

[data-md-color-scheme="default"] .vla-read-card {
  background: linear-gradient(135deg, rgba(255,255,255,.98), rgba(241,245,249,.96));
  border: 1px solid rgba(99,102,241,.16);
  box-shadow:
    0 12px 30px rgba(15,23,42,.08),
    inset 0 1px 0 rgba(255,255,255,.9);
}

[data-md-color-scheme="default"] .vla-read-body {
  background:
    radial-gradient(circle at top right, rgba(59,130,246,.14), transparent 26%),
    radial-gradient(circle at left center, rgba(139,92,246,.10), transparent 22%);
}

[data-md-color-scheme="default"] .vla-read-card h3 {
  color: #0f172a;
}

[data-md-color-scheme="default"] .vla-read-card p {
  color: #334155;
}

[data-md-color-scheme="default"] .vla-read-tag {
  color: #1d4ed8;
  background: rgba(255,255,255,.72);
  border-color: rgba(59,130,246,.16);
}

[data-md-color-scheme="default"] .vla-read-arrow {
  color: #2563eb;
}

[data-md-color-scheme="default"] .vla-read-footer {
  color: #334155;
  border-top-color: rgba(100,116,139,.16);
}

@media (max-width: 768px) {
  .vla-read-card {
    border-radius: 18px;
  }

  .vla-read-body {
    padding: 18px 16px 16px;
  }

  .vla-read-head,
  .vla-read-footer {
    align-items: flex-start;
    flex-direction: column;
  }

  .vla-read-footer strong {
    width: 100%;
    text-align: center;
  }
}
</style>
