const runButton = document.querySelector("#run-analysis");
const tabs = Array.from(document.querySelectorAll(".stage-tab"));
const panes = Array.from(document.querySelectorAll(".stage-pane"));
const suggestionsPanel = document.querySelector("#suggestions-panel");

const processItems = [
  "已读取职业健康档案文件。",
  "已完成粉尘接触史、肺功能、听力和心电图指标抽取。",
  "已根据职业暴露信息生成主检建议与随访提示。"
];

function showPane(targetId) {
  tabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.target === targetId);
  });

  panes.forEach((pane) => {
    pane.classList.toggle("active", pane.id === targetId);
  });
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => showPane(tab.dataset.target));
});

runButton.addEventListener("click", () => {
  const advicePane = document.querySelector("#advice-panel");
  const processPane = document.querySelector("#process-panel");

  advicePane.innerHTML = `
    <p>主检建议已生成。</p>
    <p>建议继续保留现有岗位健康监测，并结合肺功能结果安排复查。</p>
  `;

  processPane.innerHTML = `
    <ol class="process-list">
      ${processItems.map((item) => `<li>${item}</li>`).join("")}
    </ol>
  `;

  suggestionsPanel.classList.add("is-active");
  showPane("advice-panel");
});
