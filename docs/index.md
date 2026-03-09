---
hide:
    - date
home: true
statistics: true
comments: true
---

<section class="home-layout">
  <aside class="home-profile enter" style="--d: 0.02s;">
    <img class="home-avatar" src="https://github.com/awslasasd.png" alt="awslasasd avatar">
    <h3>awslasasd</h3>
    <p class="home-profile-desc">机器人与 AI 方向学习者，持续记录课程、项目与工具实践。</p>
    <div class="home-profile-links">
      <a href="https://github.com/awslasasd">GitHub</a>
      <a href="links/">Friends</a>
    </div>
  </aside>

  <div class="home-center">
    <section class="home-hero home-hero--v2 enter" style="--d: 0.04s;">
      <p class="home-badge">Twinkle's Notebook</p>
      <h1>
        <span id="typed-title" data-text="Learn, Build, Share">Learn, Build, Share</span>
        <span class="type-cursor" aria-hidden="true">|</span>
      </h1>
      <p class="home-subtitle">
        一个持续更新的个人知识站点：记录课程学习、机器人探索与实用工具沉淀。
      </p>
      <div class="home-actions">
        <a class="md-button md-button--primary" href="CS/DL/C_D01_Introduction/">进入笔记</a>
        <a class="md-button" href="https://github.com/awslasasd/awslasasd.github.io/commits">更新日志</a>
        <a class="md-button" href="links/">朋友们</a>
        <a class="md-button" href="javascript:toggleStatistics();">站点统计</a>
      </div>
    </section>

    <section class="home-grid home-grid--v2 enter" style="--d: 0.1s;">
      <a class="home-card" href="Class/Control/C_C01_Electrical_control/">
        <h3>Class</h3>
        <p>控制、电气、传感与机器学习课程笔记。</p>
      </a>
      <a class="home-card" href="Robotics/Navigation/R_N01_Navigation_planning/">
        <h3>Robotics</h3>
        <p>导航、定位、建模、嵌入式与多智能体。</p>
      </a>
      <a class="home-card" href="CS/ML/C_M01_ML/">
        <h3>CS / AI</h3>
        <p>机器学习、深度学习与算法学习。</p>
      </a>
      <a class="home-card" href="Tools/AI/">
        <h3>Tools</h3>
        <p>AI 工具、Git、Linux 与建站实践。</p>
      </a>
    </section>

    <div id="statistics" class="home-stats enter" style="--d: 0.14s;" aria-hidden="true">
      <p>页面总数：{{pages}}</p>
      <p>总字数：{{words}}</p>
      <p>代码行数：{{codes}}</p>
      <p>站点运行时间：<span id="web-time">加载中...</span></p>
    </div>
  </div>

  <aside class="home-recent enter" style="--d: 0.08s;">
    <h3>最近在做</h3>
    <a href="Robotics/Navigation/R_N03_Localization/">
      <span>机器人定位</span>
      <small>Navigation / Localization</small>
    </a>
    <a href="CS/DL/FishBook/C_D02_FB04_Backpropagation/">
      <span>反向传播</span>
      <small>Deep Learning / FishBook</small>
    </a>
    <a href="Tools/AI/">
      <span>AI 工具工作流</span>
      <small>Prompt + Workflow</small>
    </a>
  </aside>
</section>

<script>
function formatDuration(startDate) {
  var now = Date.now();
  var diff = now - startDate.getTime();
  var day = Math.floor(diff / (24 * 3600 * 1000));
  var hour = Math.floor(diff / (3600 * 1000) % 24);
  var minute = Math.floor(diff / (60 * 1000) % 60);
  return day + " 天 " + hour + " 小时 " + minute + " 分钟";
}

function updateTime() {
  var started = new Date("2024-06-03T09:10:00+08:00");
  var target = document.getElementById("web-time");
  target.textContent = formatDuration(started);
}

function toggleStatistics() {
  var panel = document.getElementById("statistics");
  panel.classList.toggle("is-visible");
  panel.setAttribute("aria-hidden", String(!panel.classList.contains("is-visible")));
}

function typeTitle() {
  var el = document.getElementById("typed-title");
  if (!el) return;
  var text = el.getAttribute("data-text") || "";
  var i = 0;
  el.textContent = "";
  var timer = setInterval(function () {
    el.textContent = text.slice(0, i);
    i += 1;
    if (i > text.length) clearInterval(timer);
  }, 60);
}

updateTime();
typeTitle();
document.body.classList.add("home-ready");
setInterval(updateTime, 60 * 1000);
</script>
