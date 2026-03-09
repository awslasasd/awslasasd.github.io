---
hide:
    - date
home: true
statistics: true
comments: true
---

<section class="home-hero home-hero--v2">
  <p class="home-badge">Twinkle's Notebook</p>
  <h1>Learn, Build, Share</h1>
  <p class="home-subtitle">
    一个持续更新的个人知识站点：记录课程学习、机器人探索与实用工具沉淀。
  </p>
  <div class="home-actions">
    <a class="md-button md-button--primary" href="CS/DL/C_D01_Introduction/">进入笔记</a>
    <a class="md-button" href="https://github.com/awslasasd/awslasasd.github.io/commits/main">更新日志</a>
    <a class="md-button" href="links/">朋友们</a>
    <a class="md-button" href="javascript:toggleStatistics();">站点统计</a>
  </div>
</section>

<section class="home-grid home-grid--v2">
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
    <p>机器学习、深度学习与算法学习路线。</p>
  </a>
  <a class="home-card" href="Tools/AI/">
    <h3>Tools</h3>
    <p>AI 工具、Git、Linux 与建站实践。</p>
  </a>
</section>

<div id="statistics" class="home-stats" aria-hidden="true">
  <p>页面总数：{{pages}}</p>
  <p>总字数：{{words}}</p>
  <p>代码行数：{{codes}}</p>
  <p>站点运行时间：<span id="web-time">加载中...</span></p>
</div>

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

updateTime();
setInterval(updateTime, 60 * 1000);
</script>
