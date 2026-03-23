
let isOn = false; // 摄像头控制开关
let isStreamActive = false; // 全局统一状态：标记是否有活跃流（本地视频/摄像头）
let isFrozen = false; // 冻结状态
let currentFrameId = null; // 当前冻结帧的唯一ID（用于关联结果）
let currentFrameResultIds = []; // 存储当前帧对应的所有表格行ID（用于取消冻结时批量删除）
let resultMap = new Map(); // 映射：表格行ID → 推理结果（含坐标、配置等）
let isResultVisible = false; // 结果叠加层显示状态

let currentFile = null; // 上传视频当前文件
let isUploading = false; // 防重复上传
let isPlaying = false; // 单独控制播放状态（更清晰）

let drawOn = false;// 结果画面显示

let spottingOn = false; // 是否启用字符识别

let pollingInterval = null; // 轮询定时器


let selectedTemplateId = null; // 全局变量：存储当前选中的模板ID（初始为null）

let isTrackActive = false // 是否启用跟踪


const videoPlaceholder = document.getElementById('videoPlaceholder');

const playBtn = document.getElementById('playBtn');
const stopBtn = document.getElementById('stopBtn');

const TEMPLATE_STORAGE_KEY = "detection_templates"; // 存储模板的键（本地存储用）
const spottingBtn = document.getElementById("spottingBtn");
const trackBtn = document.getElementById("trackBtn")

const videoFeed = document.getElementById('videoFeed');
const freezeCanvas = document.getElementById('freezeCanvas');
const freezeBtn = document.getElementById('freezeBtn');
const processImgBtn = document.getElementById("processImgBtn");

const toggleResultBtn = document.getElementById("toggleResultBtn");
const resultTableBody = document.getElementById("resultTableBody");
const resultOverlayCanvas = document.getElementById("resultOverlayCanvas");

const btnSaveTemplate = document.getElementById("btnSaveTemplate");
const btnLoadTemplate = document.getElementById("btnLoadTemplate");
const templateNameInput = document.getElementById("templateName");
const templateList = document.getElementById("templateList");
const drawBtn = document.getElementById("toggleDetectionBoxBtn")

const fileInput = document.getElementById('localVideo');


// 全局统计数据存储（适配4个关键指标）
const monitorData = {
  targetCount: 0,        // 检测目标数
  trackCount: 0,         // 跟踪目标数
  inferTime: 0,          // 推理耗时（ms）
  alarmCount: 0,         // 告警次数（超限目标累计）
  lastTargetCount: 0,    // 上一次检测目标数（计算变化率）
  lastTrackCount: 0,     // 上一次跟踪目标数（计算变化率）
  confidenceHistory: [], // 置信度历史数据（用于趋势图）
  maxHistoryLength: 30,  // 趋势图最大数据长度（最近30帧）
  confidenceThreshold: 0.0, // 置信度阈值（默认0.85）
  stability: 0          // 跟踪稳定性（默认92%）
};

// 图表实例存储
let accuracyChart = null;    // 检测精度趋势图
let confidenceGauge = null;  // 置信度阈值仪表盘
let stabilityGauge = null;   // 跟踪稳定性仪表盘

// 页面加载完成后初始化图表
document.addEventListener("DOMContentLoaded", function() {
  initAccuracyChart();
  initConfidenceGauge();
  initStabilityGauge();
  // 每500ms更新一次数据（平衡实时性和性能）
  setInterval(updateMonitorData, 500);
});

// 3. 定义两种状态的样式配置（便于维护）
const btnStyles = {
  // 隐藏监测框时的按钮样式（初始状态）
  hidden: {
    class: "bg-gray-700 hover:bg-gray-600 p-1.5 rounded transition-colors",
    title: "显示监测框",
    text: "显示监测框"
  },
  // 显示监测框时的按钮样式（切换后状态）
  show: {
    class: "bg-blue-600 hover:bg-blue-500 p-1.5 rounded transition-colors", // 蓝色系标识激活态
    title: "隐藏监测框",
    text: "隐藏监测框"
  }
};

// 全局状态
let globalState = {
  // 模板相关
  localTemplates: JSON.parse(localStorage.getItem(TEMPLATE_STORAGE_KEY) || "[]"),
  currentTemplate: null, // 当前加载的模板（前端缓存）
  // 跟踪相关
  trackSessionId: null, //null;  // 跟踪会话ID
  isTracking: false,
};

const drawCanvas = document.getElementById('draw-canvas');
draw_ctx = drawCanvas.getContext('2d');

/*-----------------初始化绑定-------------------*/
// 页面加载完成后初始化
window.addEventListener('load', function() {
  initTableInputListeners(); // 绑定表格输入框监听
  renderTemplateList(); // 绑定模板管理输入监听
});
/*-----------------前端状态管理-------------------*/

// // 重置为「上传成功未播放」状态（核心复用函数）
// function resetToIdleState() {

  
//   // const freezeCanvas = document.getElementById('freezeCanvas');

//   // 1. 重置全局状态
//   isPlaying = false;
//   isFrozen = false;
//   isStreamActive = false;

//   // 2. 重置视频容器
//   videoFeed.src = "";
//   videoFeed.style.display = "none";
//   freezeCanvas.classList.add('hidden');
//   videoPlaceholder.classList.remove('hidden'); // 显示「请播放」占位

//   // 按钮文本/图标重置
//   playBtn.innerHTML = '<i class="fa fa-play mr-2"></i>开始播放';
//   stopBtn.innerHTML = '<i class="fa fa-stop mr-2"></i>停止播放';


//   // 4. 重置按钮禁用状态
//   playBtn.disabled = currentFile ? false : true
//   stopBtn.disabled = true;
//   freezeBtn.disabled = true;

//   // 工业场景：可选添加日志
//   // addAlarm('视频操作', '视频播放结束，已重置状态', 'info');
// }

/*-----------------摄像头控制-------------------*/

function ctrlCamera(){
    isOn = !isOn;
    const freezeBtn = document.getElementById('freezeBtn');
    fetch("/ctrl_camera?on="+(isOn?1:0))
        .then(r=>r.json())
        .then(d=>{
            document.getElementById("cameraBtn").textContent = d.streaming? "关闭摄像头" : "打开摄像头";
            if (d.streaming){
                freezeBtn.disabled = false;
                isStreamActive = true;
                isPlaying = false; // 避免与本地视频状态冲突
                document.getElementById("videoPlaceholder").classList.add('hidden');
                document.getElementById("videoFeed").src = "";
                document.getElementById("videoFeed").src = `/video_stream?source=camera&t=${Date.now()}`;
                // document.getElementById("videoFeed").style.display = "inline"
            }else{
                freezeBtn.disabled = true;
                isStreamActive = false;
                document.getElementById('videoPlaceholder').classList.remove('hidden');
                document.getElementById("videoFeed").src = "";
                document.getElementById("videoFeed").style.display = 'none';
            }
        })
        .catch(err => console.error('[ctrlCamera] 请求失败', err));
}


/*------------------模型切换------------------------*/
async function switchModel(model) {
    const res = await fetch('/switch_model', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model})
    });
    const d = await res.json();
    alert(d.msg);          // 成功/失败提示
  }


/*------------------视频上传与播放管理--------------------------*/

// 选择文件后自动上传+上传成功后可播放
async function uploadAndPlayVideo() {
  const file = fileInput.files[0];
  const uploadBtn = fileInput.previousElementSibling; 
  // 前置校验
  if (isUploading) return alert("正在上传中，请稍候...");
  if (!file) return;
  if (!file.type.startsWith('video/')) {
    fileInput.value = '';
    return alert("请选择有效的视频文件（MP4/AVI/MOV等）");
  }

  // 上传中状态
  isUploading = true;
  uploadBtn.disabled = true;
  uploadBtn.innerHTML = '<i class="fa fa-spinner fa-spin mr-2"></i>上传中...';
  // 初始禁用播放/停止按钮
  // playBtn.disabled = true;
  // stopBtn.disabled = true;
  // document.getElementById('playBtn').disabled = true;
  // document.getElementById('stopBtn').disabled = true;

  try {
    // 上传视频
    const fd = new FormData();
    fd.append("video", file);
    const res = await fetch("/upload_video", {
      method: "POST",
      body: fd,
      timeout: 30000
    });

    if (!res.ok) throw new Error(`服务器错误：${res.status}`);
    const d = await res.json();
    if (d.code !== 0) throw new Error(d.msg || "上传失败");

    // 上传成功：保存文件+启用播放按钮
    currentFile = d.file;
    // playBtn.disabled = false;
    alert(`✅ 上传成功！\n文件名：${file.name}\n点击「开始播放」按钮启动视频`);

  } catch (err) {
    alert(`❌ 操作失败：${err.message}`);
    console.error('详细错误：', err);
  } finally {
    isUploading = false;
    uploadBtn.disabled = false;
    uploadBtn.innerHTML = '<i class="fa fa-file-video-o mr-2"></i>选择本地视频';
    fileInput.value = '';
    // startPlayback();
  }
}

// 单独的「开始/暂停播放」函数
async function startPlayback() {
  if (!currentFile) return alert("无可用视频，请先选择文件上传");
  if (isPlaying) {
    playBtn.innerHTML = '<i class="fa fa-spinner fa-spin mr-2"></i>暂停中...';
    const res = await fetch(`/ctrl_video?video_state=pause`, {timeout: 1000});
    isPlaying = false;
  }else{
    // 播放中状态
    document.getElementById("videoPlaceholder").classList.add('hidden');
    // playBtn.disabled = true;
    playBtn.innerHTML = '<i class="fa fa-spinner fa-spin mr-2"></i>播放中...';

    try {
      const res = await fetch(`/ctrl_video?video_state=play`, { timeout: 1000 });
      const d = await res.json();
      if (d.streaming) {
        isPlaying = true;
        isStreamActive = true;
        videoFeed.src = `/video_stream?source=file&file=${currentFile}`;
        videoFeed.style.display = "inline";
        // 启用停止按钮，禁用播放按钮
        // stopBtn.disabled = false;
        // freezeBtn.disabled = false; // 播放时启用冻结按钮
      }
    } catch (err) {
      alert(`❌ 播放失败：${err.message}`);
      document.getElementById("videoPlaceholder").classList.remove('hidden');
      // playBtn.disabled = false;
      playBtn.innerHTML = '<i class="fa fa-play mr-2"></i>开始播放';
    }

  }
}

// 单独的「结束播放」函数（彻底停止循环/非循环播放）
async function stopPlayback() {

    // 停止中状态（优化反馈：图标+文字）
    // stopBtn.disabled = true;
    stopBtn.innerHTML = '<i class="fa fa-spinner fa-spin mr-2"></i>停止中...';
  
    try {
      // 1. 通知后端停止流推送（关键：终止后端循环播放逻辑）
      const res = await fetch(`/ctrl_playing?on=0`);
      if (!res.ok) throw new Error(`停止失败：${res.status}（服务器错误）`);

      // 2. 前端强制清空流（避免残留播放）
      // videoFeed.src = "";
      // videoFeed.load(); // 强制刷新视频元素
      isStreamActive = false;
      videoFeed.src = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==";
      videoPlaceholder.classList.remove('hidden');
      // 3. 按钮状态重置
      isPlaying = false;
      playBtn.innerHTML = '<i class="fa fa-play mr-2"></i>开始播放';
      stopBtn.innerHTML = '<i class="fa fa-stop mr-2"></i>停止播放';
      // videoFeed.src = "";
      // videoFeed.style.display = "none";
      // resetToIdleState();
    } catch (err) {
      // 错误处理：恢复按钮状态+提示
      alert(`❌ 停止失败：${err.message}`);
      console.error('[结束播放] 详细错误：', err);
      // stopBtn.disabled = false;
      stopBtn.innerHTML = '<i class="fa fa-stop mr-2"></i>停止播放';
    }
  }

// // 🔴 修复：监听<img>标签的错误事件（适配MJPEG流中断）
// videoFeed.addEventListener('error', () => {
//   console.warn('视频流异常中断（可能已停止或播放完毕）');
//   // 仅在播放中状态时重置（避免初始加载时误触发）
//   if (isPlaying) {
//     resetToIdleState();
//     alert("视频播放已停止或异常中断");
//   }
// });

/*----------------------------------------------------冻结帧-------------------------------------------------------------------------------*/
// 冻结/解冻 切换函数
function ctrlFreezeFrame() { 
    // 关键修复：判断统一状态 isStreamActive，而非单独判断 isPlaying/isOn
    if (!isStreamActive) {
      return alert("无活跃流（未播放本地视频/未启动摄像头），无法冻结帧");
    }
  
    if (!isFrozen) {
      // 👉 冻结帧：捕获当前画面并显示
      isFrozen = true;
      // 1. 设置画布尺寸与视频容器一致
      const container = videoFeed.parentElement;
      freezeCanvas.width = container.clientWidth;
      freezeCanvas.height = container.clientHeight;
      resultOverlayCanvas.width = container.clientWidth;
      resultOverlayCanvas.height = container.clientHeight;

      // 2. 复制当前视频流画面到画布
      const ctx = freezeCanvas.getContext('2d');
      ctx.drawImage(videoFeed, 0, 0, freezeCanvas.width, freezeCanvas.height);
      // 3. 显示画布、隐藏原视频流
      freezeCanvas.classList.remove('hidden');
      videoFeed.classList.add('hidden');
      videoFeed.style.display = 'none';

      currentFrameId = `frame_${Date.now()}`; // 生成唯一帧ID
      currentFrameResultIds = []; // 初始化结果行ID列表
      // 4. 更新按钮状态
      // processImgBtn.disabled = false;
      freezeBtn.innerHTML = '<i class="fa fa-play-circle mr-2"></i>解冻';
      freezeBtn.classList.remove('bg-industrial-primary/70', 'hover:bg-industrial-primary');
      freezeBtn.classList.add('bg-industrial-info/70', 'hover:bg-industrial-info');
      // toggleResultBtn.disabled = false;
      // 5. 禁用播放/停止按钮（避免状态冲突）
      // document.getElementById('playBtn').disabled = true;
      // document.getElementById('stopBtn').disabled = true;
      // 工业场景：添加操作日志
    //   addAlarm('视频操作', '已冻结当前帧', 'info');
    } else {
      // 👉 解冻：恢复原视频流播放
      isFrozen = false;
      // 1. 隐藏画布、显示原视频流
      freezeCanvas.classList.add('hidden');
      resultOverlayCanvas.classList.add('hidden'); // 隐藏叠加层
      videoFeed.style.display = 'inline';
      // 2. 循环删除所有关联的表格行
      currentFrameResultIds.forEach(rowId => {
        const row = document.getElementById(rowId);
        if (row) row.remove();
      });
      // 若表格为空，恢复空状态
      if (resultTableBody.children.length === 0) {
        resultTableBody.innerHTML = `
            <tr>
                <td colspan="8" class="py-8 text-center text-gray-500">
                    <i class="fa fa-search-minus text-2xl mb-2"></i>
                    <p>暂无推理结果，请先执行推理</p>
                </td>
            </tr>
        `;
      }
      // 清空状态
      currentFrameId = null;
      currentFrameResultIds = [];
      resultMap.clear();
      // 3. 更新按钮状态
      // processImgBtn.disabled = true;
      freezeBtn.innerHTML = '<i class="fa fa-pause-circle mr-2"></i>冻结帧';
      freezeBtn.classList.remove('bg-industrial-info/70', 'hover:bg-industrial-info');
      freezeBtn.classList.add('bg-industrial-primary/70', 'hover:bg-industrial-primary');
      // document.getElementById('playBtn').disabled = false;
      // document.getElementById('stopBtn').disabled = false;
      // toggleResultBtn.disabled = true;
  
      // 工业场景：添加操作日志
    //   addAlarm('视频操作', '已解冻视频播放', 'info');
    }
  }

/*-----------------独立处理图像函数（上传后端+渲染结果）-------------------*/
async function processFrozenImage() {
  if (!isFrozen) {
      return alert("请先冻结帧，再进行处理");
  }
  if (!currentFrameId) {
      return alert("帧ID异常，无法处理");
  }

  // 处理中状态：禁用按钮+显示加载中
  // processImgBtn.disabled = true;
  processImgBtn.innerHTML = '<i class="fa fa-spinner fa-spin mr-1"></i>处理中...';
  currentFrameResultIds = []; // 清空历史结果行ID

  try {
      // 1. Canvas 转 Base64（原有逻辑）
      const base64Img = freezeCanvas.toDataURL('image/jpeg', 0.8);
      const pureBase64 = base64Img.replace('data:image/jpeg;base64,', '');

      // 2. 上传到后端（携带帧ID和流源类型）
      const res = await fetch("/process_freeze_frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
              image_base64: pureBase64,
              frame_id: currentFrameId,
              source: isOn ? "camera" : "video" // isOn 是摄像头状态变量（你的原有变量）
          }),
          timeout: 15000 // 延长超时时间（适配复杂处理）
      });

      if (!res.ok) throw new Error(`服务器处理失败：${res.status}`);
      const data = await res.json();

      // 3. 关键：判断结果是否为列表
      if (!Array.isArray(data.results)) {
          throw new Error("后端返回结果格式错误，应为List[dict]");
      }

      const results = data.results;
      if (results.length === 0) {
        // 无结果时渲染空提示行
        renderEmptyResultRow();
        return;
      }
      // 4. 循环渲染列表中的每个结果（1个元素=1行表格）
      results.forEach((result, index) => {
          // 生成唯一行ID：帧ID+索引（确保同一帧多个结果不重复）
          const rowId = `${currentFrameId}_${index}`;
          currentFrameResultIds.push(rowId); // 记录行ID，用于后续删除
          renderSingleResultToTable(rowId, result, index + 1); // 渲染单行
          // checkAndTriggerAlarm(result); // 逐行检查告警
      });
  } catch (err) {
      console.error("图像处理失败：", err);
      // 渲染错误信息到表格
      renderErrorResultRow(err.message);
  } finally {
      // 恢复按钮状态
      processImgBtn.disabled = false;
      processImgBtn.innerHTML = '<i class="fa fa-cogs mr-1"></i>处理图像';
  }
}

/**
 * 渲染单个结果到表格（1个dict=1行）
 * @param {string} rowId - 表格行唯一ID
 * @param {Object} result - 单个结果dict
 * @param {number} seq - 序号（用于推理ID后缀）
 */
function renderSingleResultToTable(rowId, result, seq) {
  // 清空表格空状态
  if (resultTableBody.querySelector('td[colspan="8"]')) {
      resultTableBody.innerHTML = '';
  }

  const timeStr = new Date().toLocaleString('zh-CN', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit'
  });

  const row = document.createElement('tr');
  row.id = rowId;
  row.className = "border-b border-gray-700 hover:bg-industrial-darker/50";
  row.innerHTML = `
      <td class="py-3 px-4">
          <input type="checkbox" class="result-checkbox" value="${rowId}">
      </td>
      <td class="py-3 px-4 font-medium">${currentFrameId}_${seq}</td> <!-- 推理ID：帧ID+序号 -->
      <td class="py-3 px-4">
        <input type="number" 
        value="${seq}"
        class="track-id w-24 bg-industrial-darker border border-gray-700 rounded-md px-2 py-1 text-sm"
        placeholder="输入跟踪id"
        data-rowid="${rowId}" <!-- 关联 rowId -->
        >
      </td>
      <td class="py-3 px-4">${result.text}</td>
      <td class="py-3 px-4">
        <input type="text" 
        value="${result.obj}"
        class="user-label w-24 bg-industrial-darker border border-gray-700 rounded-md px-2 py-1 text-sm"
        placeholder="输入标签"
        data-rowid="${rowId}" <!-- 关联 rowId -->
        >
      </td>
      <td class="py-3 px-4">
          <input 
          type="number" 
          placeholder="上限" 
          class="num-limit upper-limit w-16 bg-industrial-darker border border-gray-700 rounded-md px-2 py-1 text-sm" 
          value="${result.upper_limit || 20}"
          data-rowid="${rowId}" <!-- 关联 rowId -->
          > 
      </td>
      <td class="py-3 px-4">
          <input 
          type="number" 
          placeholder="下限" 
          class="num-limit lower-limit w-16 bg-industrial-darker border border-gray-700 rounded-md px-2 py-1 text-sm" 
          value="${result.lower_limit || 0}"
          data-rowid="${rowId}" <!-- 关联 rowId -->
          >
      </td>
      <td class="py-3 px-4">
          <button class="text-gray-400 hover:text-gray-300" onclick="deleteResult('${rowId}')">
              <i class="fa fa-trash"></i> 删除
          </button>
      </td>
  `;
  resultTableBody.appendChild(row);
  // 更新结果映射（用于叠加层绘制）
  resultMap.set(rowId, {
    trackId: seq,
    text: result.text, // 初始用户标号为text
    userLabel: result.obj,  // 初始用户标号为默认text
    upperLimit: result.upper_limit || 20,
    lowerLimit: result.lower_limit || 0,
    bbox: result.bbox,  // 后端处理后的 [x1, y1, x2, y2]
    confidence: result.confidence,
    rowid: rowId
});

}

/**
 * 渲染无结果提示行
 */
function renderEmptyResultRow() {
  if (resultTableBody.querySelector('td[colspan="8"]')) {
      resultTableBody.innerHTML = '';
  }

  const row = document.createElement('tr');
  row.innerHTML = `
      <td class="py-8 text-center text-gray-500" colspan="8">
          <i class="fa fa-search-minus text-2xl mb-2"></i>
          <p>当前冻结帧未检测到任何目标</p>
      </td>
  `;
  resultTableBody.appendChild(row);
}

/**
 * 渲染错误结果行
 * @param {string} errorMsg - 错误信息
 */
function renderErrorResultRow(errorMsg) {
  if (resultTableBody.querySelector('td[colspan="8"]')) {
      resultTableBody.innerHTML = '';
  }

  const row = document.createElement('tr');
  row.className = "border-b border-gray-700 text-industrial-danger";
  row.innerHTML = `
      <td class="py-3 px-4" colspan="8">
          <div class="flex items-center">
              <i class="fa fa-exclamation-circle mr-2"></i>
              <span>帧ID:${currentFrameId} - 处理失败：${errorMsg}</span>
          </div>
      </td>
  `;
  resultTableBody.appendChild(row);
}

/* 冻结画面同步显示*/
function bindToggleResultBtn() {
    isResultVisible = !isResultVisible;
    if (isResultVisible) {
        toggleResultBtn.innerHTML = '<i class="fa fa-eye-slash mr-1"></i>隐藏结果';
        resultOverlayCanvas.classList.remove('hidden');         // 显示叠加层
        redrawResultOverlay(); // 重新绘制（确保同步最新表格编辑）
    } else {
        toggleResultBtn.innerHTML = '<i class="fa fa-eye mr-1"></i>显示结果';
        resultOverlayCanvas.classList.add('hidden'); // 隐藏叠加层
    }
}

// 工具函数：从字符串中提取第一个连续数值（支持整数、小数、负数）
function extractFirstNumber(str) {
  // 正则匹配：-?\d+(\.\d+)? → 匹配负数、整数、小数（连续数字部分）
  const numMatch = str.match(/-?\d+(\.\d+)?/);
  return numMatch ? Number(numMatch[0]) : null; // 提取并转为数字，无匹配返回null
}

function redrawResultOverlay() {
  toggleResultBtn.innerHTML = '<i class="fa fa-eye-slash mr-1"></i>隐藏结果';
  const ctx = resultOverlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, resultOverlayCanvas.width, resultOverlayCanvas.height);

  resultMap.forEach(result => {
    try{
      const {text, userLabel, upperLimit, lowerLimit, confidence, bbox, rowid } = result;
      
      const validPoints = [];
      let isBboxValid = true;

      // 遍历所有顶点（不再硬限制4个，适配任意长度）
      for (let i = 0; i < bbox.length; i++) {
        const point = bbox[i];
        // 校验点的格式（必须是长度为2的数组）
        if (!Array.isArray(point) || point.length !== 2) {
          isBboxValid = false;
          break;
        }
        // 转换为数字并校验有效性（非NaN、非负、有限值）
        const x = Number(point[0]);
        const y = Number(point[1]);
        if (isNaN(x) || isNaN(y) || x < 0 || y < 0 || !isFinite(x) || !isFinite(y)) {
          isBboxValid = false;
          break;
        }
        validPoints.push([x, y]);
      }

      // 4.2 应用坐标变换（原始坐标 → 显示坐标，适配任意顶点数）
      const { scale, offsetX, offsetY } = getVideoTransform(1280, 720);
      const transformedPoints = validPoints.map(([x, y]) => [
        x * scale + offsetX, // 缩放+水平偏移（适配左黑边）
        y * scale + offsetY  // 缩放+垂直偏移（适配上黑边）
      ]);

      // 计算目标框坐标范围、中心、宽高（兼容任意多边形）
      const xCoords = transformedPoints.map(p => p[0]);
      const yCoords = transformedPoints.map(p => p[1]);
      // const xCoords = bbox.map(point => Number(point[0]));
      // const yCoords = bbox.map(point => Number(point[1]));
      const minX = Math.min(...xCoords);
      const maxX = Math.max(...xCoords);
      const minY = Math.min(...yCoords);
      const maxY = Math.max(...yCoords);
      const centerX = (minX + maxX) / 2;  // 多边形外接矩形中心
      const centerY = (minY + maxY) / 2;
      const boxWidth = maxX - minX;       // 外接矩形宽度
      const boxHeight = maxY - minY;      // 外接矩形高度
      const minBoxSize = 8; // 最小可见尺寸（外接矩形的最小宽高）

      // 1. 确定边框颜色（醒目色，逻辑不变）
      let borderColor = '#00ff00'; // 正常绿色
      // 核心判断逻辑
      const extractedNum = extractFirstNumber(text); // 提取字符串中的连续数值
      const isLabelMatch = text === userLabel; // 字符串是否完全等于userLabel
      const isNumInRange = extractedNum !== null && !isNaN(extractedNum) && extractedNum >= lowerLimit && extractedNum <= upperLimit; // 数值在上下限内

      // 两个条件都不满足 → 视为超限（红色边框）
      if (!isLabelMatch && !isNumInRange) {
        borderColor = '#ff0000'; // 超限红色
      }
      // const text_num = Number(text);
      // if (text !== userLabel && (Number.isNaN(text_num) || text_num > upperLimit || text_num < lowerLimit)) {
      //   borderColor = '#ff0000'; // 超限红色
      // }

      // 2. 调整过小的目标框（关键修改：适配多边形）
      let adjustedBbox = transformedPoints;
      // let adjustedBbox = bbox;
      if (boxWidth < minBoxSize || boxHeight < minBoxSize) {
        // 过小多边形 → 替换为「最小尺寸的正方形」（保持中心不变，避免绘制不可见）
        adjustedBbox = [
          [centerX - minBoxSize/2, centerY - minBoxSize/2], // 左上
          [centerX + minBoxSize/2, centerY - minBoxSize/2], // 右上
          [centerX + minBoxSize/2, centerY + minBoxSize/2], // 右下
          [centerX - minBoxSize/2, centerY + minBoxSize/2]  // 左下
        ];
      }

      // 3. 绘制目标框（兼容任意多边形：循环遍历所有顶点）
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 3;
      ctx.beginPath();
      // 移动到第一个顶点（起点）
      ctx.moveTo(adjustedBbox[0][0], adjustedBbox[0][1]);
      // 循环遍历剩余所有顶点（自动适配任意顶点数）
      for (let i = 1; i < adjustedBbox.length; i++) {
        const [x, y] = adjustedBbox[i];
        ctx.lineTo(x, y);
      }
      // 闭合路径（自动连接最后一个顶点到起点）
      ctx.closePath();
      ctx.stroke();

      // 半透明填充（突出目标区域）
      ctx.fillStyle = borderColor + '20';
      ctx.fill();

      // 4. 绘制中文文本（框上方，避免遮挡）
      ctx.font = "14px Arial, 'Microsoft YaHei'";
      const displayText = text || "未知目标";
      const textWidth = ctx.measureText(displayText).width;
      const textHeight = 20;
      // 文本背景
      const bgX = centerX - textWidth / 2 - 4;
      const bgY = Math.min(...yCoords) - textHeight - 2; // 框上方5px
      ctx.fillStyle = borderColor + '80';
      ctx.fillRect(bgX, bgY, textWidth + 8, textHeight);
      // 文本内容
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(displayText, centerX, bgY + textHeight/2);

      // // 5. 绘制置信度（使用上面修复后的代码）
      // ctx.font = "12px Arial";
      // const validConfidence = typeof confidence === 'number' && !isNaN(confidence) ? confidence : 0.00;
      // const confText = `置信度: ${validConfidence.toFixed(2)}`;
      // const confWidth = ctx.measureText(confText).width;
      // const confBgX = centerX - confWidth / 2 - 4;
      // const confBgY = Math.max(...yCoords) + 5;

      // const canvasHeight = resultOverlayCanvas.height;
      // const confBgHeight = 16;
      // let finalConfBgY = confBgY;
      // if (finalConfBgY + confBgHeight > canvasHeight) {
      //     finalConfBgY = Math.min(...yCoords) - confBgHeight - 5;
      // }

      // ctx.fillStyle = borderColor + '80';
      // ctx.fillRect(confBgX, finalConfBgY, confWidth + 8, confBgHeight);
      // ctx.fillStyle = '#ffffff';
      // ctx.textAlign = 'center';
      // ctx.fillText(confText, centerX, finalConfBgY + 8);

    } catch (err) {
      console.error("绘制失败：", err);
      // 报错后不中断，继续绘制下一个目标
    }
  });
}

/*-----------------删除表格行（同步删除映射+叠加层）-------------------*/
function deleteResult(rowId) {
  // 删除表格行
  const row = document.getElementById(rowId);
  if (row) row.remove();
  // 删除结果映射
  resultMap.delete(rowId);
  // 更新结果行ID列表
  currentFrameResultIds = currentFrameResultIds.filter(id => id !== rowId);
  // 重新绘制叠加层（移除已删除目标）
  if (isResultVisible) {
      redrawResultOverlay();
  }
  // 表格为空时恢复空状态
  if (resultTableBody.children.length === 0) {
      resultTableBody.innerHTML = `
          <tr>
              <td colspan="8" class="py-8 text-center text-gray-500">
                  <i class="fa fa-search-minus text-2xl mb-2"></i>
                  <p>暂无推理结果，请先执行推理</p>
              </td>
          </tr>
      `;
      toggleResultBtn.disabled = true;
      resultOverlayCanvas.classList.add('hidden');
  }
}


/*-----------------初始化：绑定表格输入框监听（实时同步）-------------------*/
function initTableInputListeners() {
  // 监听表格中所有输入框变化（事件委托，支持动态新增行）
  resultTableBody.addEventListener('input', function(e) {
      const target = e.target;
      // 1. 监听「用户标号」输入框
      if (target.classList.contains('user-label')) {
          const rowId = target.dataset.rowid; // 获取关联的 rowId
          const result = resultMap.get(rowId); // 从 resultMap 中获取对应结果
          if (!result) return; // 未找到结果，直接返回
          // 同步修改后的值到 resultMap
          const newUserLabel = target.value;
          result.userLabel = newUserLabel;
      }
      else if (target.classList.contains('track-id')) {
        const rowId = target.dataset.rowid; // 获取关联的 rowId
        const result = resultMap.get(rowId); // 从 resultMap 中获取对应结果
        if (!result) return; // 未找到结果，直接返回
        // 同步修改后的值到 resultMap
        const newUserLabel = target.value;
        result.trackId = newUserLabel;
      }
      // 2. 监听「数值上下限」输入框
      else if (target.classList.contains('num-limit')) {
          const rowId = target.dataset.rowid; // 获取关联的 rowId
          const result = resultMap.get(rowId); // 从 resultMap 中获取对应结果
          if (!result) return; // 未找到结果，直接返回

          // 同步修改后的值到 resultMap
          if (target.classList.contains('upper-limit')) {
              const newUpperLimit = Number(target.value);
              // 容错：确保是有效数字，否则用默认值 20
              result.upperLimit = isNaN(newUpperLimit) ? 20 : newUpperLimit;
          } else if (target.classList.contains('lower-limit')) {
              const newLowerLimit = Number(target.value);
              // 容错：确保是有效数字，否则用默认值 0
              result.lowerLimit = isNaN(newLowerLimit) ? 0 : newLowerLimit;
          }
          // 关键：重新绘制叠加层，更新边框颜色
          redrawResultOverlay();
      }
  });
}

/*------------------------------------------模板保存与加载功能管理---------------------------------------*/
function saveTemplateBtn(){
  const templateName = templateNameInput.value.trim();
  if (!templateName) {
    return alert("请输入模板名称");
  }
  if (resultMap.size === 0) {
    return alert("当前无检测结果，无法保存模板");
  }

  // 转换 resultMap 为数组（Map 无法直接存储，需转数组）
  const templateContent = Array.from(resultMap.values()).map(item => ({
    rowId: item.rowId || "",
    text: item.text || "",
    userLabel: item.userLabel || "",
    trackId: item.trackId || "",
    upperLimit: item.upperLimit || 20,
    lowerLimit: item.lowerLimit || 0,
    confidence: item.confidence || 0,
    bbox: item.bbox || [[10,10], [100,10], [100,100], [10,100]]
  }));
  // 构建模板对象
  const newTemplate = {
    templateId: `template_${Date.now()}_${Math.floor(Math.random() * 1000)}`, // 唯一ID
    templateName: templateName,
    createTime: new Date().toLocaleString('zh-CN', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    }),
    content: templateContent
  };
  // 从本地存储获取已有模板，添加新模板
  const existingTemplates = JSON.parse(localStorage.getItem(TEMPLATE_STORAGE_KEY) || "[]");
  existingTemplates.push(newTemplate);
  localStorage.setItem(TEMPLATE_STORAGE_KEY, JSON.stringify(existingTemplates));

  // 刷新模板列表
  renderTemplateList();
  // 清空输入框
  templateNameInput.value = "";
  alert(`模板「${templateName}」保存成功`);
}

// 渲染已保存的模板列表
function renderTemplateList() {
  const existingTemplates = JSON.parse(localStorage.getItem(TEMPLATE_STORAGE_KEY) || "[]");

  if (existingTemplates.length === 0) {
    templateList.innerHTML = '<div class="text-gray-500 text-center py-2">暂无保存的模板</div>';
    return;
  }

  // 渲染模板列表（按保存时间倒序）
  const sortedTemplates = existingTemplates.sort((a, b) => 
    new Date(b.createTime) - new Date(a.createTime)
  );

  templateList.innerHTML = sortedTemplates.map(template => `
    <div 
    class="bg-industrial-darker rounded-md p-2 flex justify-between items-center hover:bg-gray-700 transition-colors cursor-pointer"
    onclick="selectTemplate('${template.templateId}')"
    id="template-row-${template.templateId}" // 给每行添加唯一ID，用于控制选中样式
    >
      <div>
        <p class="font-medium text-gray-400">${template.templateName}</p>
        <p class="text-xs text-gray-400">${template.createTime} · 目标数：${template.content.length}</p>
      </div>
      <div class="flex gap-1">
        <button 
        class="text-xs text-industrial-primary hover:text-industrial-primary/80" 
        onclick="loadTemplate('${template.templateId}')"
        >
          <i class="fa fa-download mr-1"></i>加载
        </button>
        <button 
        class="text-xs text-gray-400 hover:text-gray-300" 
        onclick="deleteTemplate('${template.templateId}')"
        >
          <i class="fa fa-trash mr-1"></i>删除
        </button>
      </div>
    </div>
  `).join("");
}

// 加载模板（全局函数，模板列表按钮调用）
function loadTemplate(templateId) {
  const existingTemplates = JSON.parse(localStorage.getItem(TEMPLATE_STORAGE_KEY) || "[]");
  const targetTemplate = existingTemplates.find(t => t.templateId === templateId);

  if (!targetTemplate) {
    return alert("模板不存在或已被删除");
  }

  // 清空当前结果（表格+resultMap）
  currentFrameResultIds.forEach(rowId => {
    const row = document.getElementById(rowId);
    if (row) row.remove();
  });
  resultMap.clear();
  currentFrameResultIds = [];

  // 渲染模板中的目标到表格，并更新 resultMap
  targetTemplate.content.forEach((item, index) => {
    const rowId = `template_${templateId}_${index}`; // 生成唯一行ID
    currentFrameResultIds.push(rowId);
    // 调用表格渲染函数
    renderSingleResultToTable(rowId, {
      obj: item.userLabel,
      text: item.text,
      track_id: item.trackId,
      upper_limit: item.upperLimit,
      lower_limit: item.lowerLimit,
      confidence: item.confidence,
      bbox: item.bbox
    }, index + 1);

    // 更新 resultMap
    resultMap.set(rowId, {
      text: item.text || "",
      userLabel: item.userLabel || "",
      trackId: item.trackId || `track_${index + 1}`,
      upperLimit: item.upperLimit || 20,
      lowerLimit: item.lowerLimit || 0,
      confidence: item.confidence || 0,
      bbox: item.bbox || [[10,10], [100,10], [100,100], [10,100]]
    });
  });

  // 重新绘制叠加层（显示模板中的目标框）
  redrawResultOverlay();
  alert(`模板「${targetTemplate.templateName}」加载成功`);

  // （可选）后续对接后端：通知后端当前加载的模板，用于字符匹配/跟踪
  // notifyBackendLoadTemplate(targetTemplate);
}

// 删除模板（全局函数，模板列表按钮调用）
function deleteTemplate(templateId) {
  if (!confirm("确定要删除该模板吗？删除后不可恢复")) {
    return;
  }

  const existingTemplates = JSON.parse(localStorage.getItem(TEMPLATE_STORAGE_KEY) || "[]");
  // 过滤掉要删除的模板
  const updatedTemplates = existingTemplates.filter(t => t.templateId !== templateId);
  localStorage.setItem(TEMPLATE_STORAGE_KEY, JSON.stringify(updatedTemplates));

  // 刷新模板列表
  renderTemplateList();
  alert("模板删除成功");
}
/*------------------------------------------后端模板匹配跟踪---------------------------------------*/

/**
 * 选中模板行（点击模板行时触发）
 * @param {string} templateId - 被选中模板的ID
 */
function selectTemplate(templateId) {
  // 1. 移除所有模板行的选中样式
  document.querySelectorAll("[id^='template-row-']").forEach(row => {
      row.classList.remove("template-row-selected");
  });

  // 2. 给当前选中行添加选中样式
  const selectedRow = document.getElementById(`template-row-${templateId}`);
  if (selectedRow) {
      selectedRow.classList.add("template-row-selected");
  }

  // 3. 记录选中的模板ID
  selectedTemplateId = templateId;

  // 4. 启用/禁用独立加载按钮（选中模板后启用）
  // document.getElementById("load-selected-btn").disabled = !selectedTemplateId;
}


// 加载本地模板并发送给后端
async function loadTemplateBtn() {
  // 1. 校验是否选中模板
  if (!selectedTemplateId) {
    alert("请先在模板列表中选中一个模板！");
    return;
  }
  // 2. 根据选中的templateId，从本地模板列表中匹配模板
  globalState.localTemplates = JSON.parse(localStorage.getItem(TEMPLATE_STORAGE_KEY) || "[]");
  const targetTemplate = globalState.localTemplates.find(
    template => template.templateId === selectedTemplateId
  );
  if (!targetTemplate) {
    alert("模板不存在或已删除");
    return;
  }
  globalState.currentTemplate = targetTemplate;

  try {
    // 2. 发送模板到后端，初始化跟踪会话
    const res = await fetch("/api/track/init", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        template: targetTemplate, // 发送完整模板数据
        clientId: "frontend_" + Date.now() // 前端唯一标识
      }),
      timeout: 5000
    });

    if (!res.ok) throw new Error(`后端初始化失败：${res.status}`);
    const data = await res.json();
    globalState.trackSessionId = data.sessionId; // 后端返回的会话ID
    globalState.isTracking = true;

    // 3. 前端渲染模板目标（表格+画布）
    // renderTemplateTargets(targetTemplate.content);
    alert(`模板「${targetTemplate.templateName}」加载成功，可启动后端跟踪`);

  } catch (err) {
    console.error("加载并发送模板失败：", err);
    alert("模板加载失败，请检查网络或后端服务");
    globalState.isTracking = false;
  }
}


/*-----------------字符识别功能控制-------------------*/

function ctrlSpotting(){
  spottingOn = !spottingOn;
  fetch("/ctrl_spotting?on="+(spottingOn?1:0))
      .then(r=>r.json())
      .then(d=>{
          spottingBtn.textContent = d.spotting? "识别中..." : "启动识别";
          if (d.spotting) {
            startPollingAndDrawing();
          } else {
            stopSpottingAndTrack();
            ctrlTrack();
          }
      })
      .catch(err => console.error("识别开关失败", err));
}

/*-----------------字符跟踪功能控制-------------------*/
function ctrlTrack(){
  if (!globalState.isTracking)return alert(`❌ 操作失败: 未载入模板`);
  isTrackActive = !isTrackActive;
  fetch("/ctrl_tracking?on="+(isTrackActive?1:0))
        .then(r=>r.json())
        .then(d=>{
            trackBtn.textContent = d.tracking? "跟踪中" : "启动跟踪";
            if (!d.tracking){
              stopTrack();
            }
        })
        .catch(err => console.error("跟踪开关失败", err))
}

/*----------------显示检测框----------------------*/
function toggleDetectionBox() {
  drawOn = !drawOn;
  fetch('/ctrl_draw?on=' + (drawOn?1:0))
    .then(r => r.json())
    .then(d => {
      const targetStyle = d.draw ? btnStyles.show : btnStyles.hidden;
      drawBtn.className = targetStyle.class; // 替换完整 Tailwind 类
      drawBtn.title = targetStyle.title;     // 更新悬停提示
    });
}


// 启用跟踪推理或者仅识别(后端结果接收)
function startPollingAndDrawing(){
      // 每30ms轮询一次（接近实时，不卡顿）
      pollingInterval = setInterval(async () => {
      try {
          // 调用后端接口获取跟踪结果
          const res = await fetch(`/api/track/results?sessionId=${globalState.trackSessionId}`);
          const data = await res.json();
          if (data.success && data.trackedTargets.length > 0 && drawOn) {
              // 清除上一帧绘制内容
              clearCanvas();
              // 绘制当前帧跟踪（识别）结果
              drawTrackedTargets(data.trackedTargets, data.width, data.height);
              setMonitorSource(data.trackedTargets, data.spotting_nums);
          } else {
              // 无结果时清除画布
              clearCanvas();
          }
      } catch (err) {
          console.error("获取跟踪结果失败：", err);
          clearCanvas();
      }
    }, 300);
}

// 关闭字符识别和跟踪推理
async function  stopSpottingAndTrack() {
  try {
      // 清除定时器和画布
      clearInterval(pollingInterval);
      clearCanvas();
      if (!globalState.trackSessionId) return;
      // if (!isTrackActive){
      //  stopTrack(); 
      // }
  } catch (err) {
      console.error("停止跟踪失败：", err);
      alert("停止跟踪失败，请重试");
  }
}
// 关闭跟踪推理
async function  stopTrack() {
  try {
      if (!globalState.trackSessionId) return;
      // 调用后端接口停止跟踪会话
      await fetch("/api/track/stop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessionId: globalState.trackSessionId })
      });
      // 重置状态
      globalState.trackSessionId = null;
      alert("跟踪已停止！");
  } catch (err) {
      console.error("停止跟踪失败：", err);
      alert("停止跟踪失败，请重试");
  }
}

/**
 * 清除Canvas画布
 */
function clearCanvas() {
  draw_ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
}

// 全局变量（确保已初始化）

// 初始化绘制上下文（在页面加载完成后调用）
function initDrawCanvas() {
    if (!drawCanvas) {
        console.error("未找到 draw-canvas 元素！");
        return;
    }
    draw_ctx = drawCanvas.getContext('2d');
    // 同步 Canvas 尺寸与容器一致（初始加载时）
    syncCanvasSize();
    console.log("绘制上下文初始化完成");
}


// 计算视频流 object-contain 后的变换参数（核心逻辑）
function getVideoTransform(w, h) {
    // const videoFeed = document.getElementById('videoFeed');
    // const container = videoFeed
    const container = videoFeed.parentElement;
    // const container = document.querySelector('.aspect-video');

    // 容错：容器/视频流未加载
    if (!container) {
        console.warn("视频流或容器未就绪，无法计算变换参数");
        return { scale: 1, offsetX: 0, offsetY: 0 };
    }

    // 关键参数：视频原始分辨率（后端 bbox 对应的分辨率，必须与后端一致！）
    // ********** 请根据实际后端配置修改 **********
    const VIDEO_ORIGIN_WIDTH = w;  // 后端摄像头/视频原始宽度
    const VIDEO_ORIGIN_HEIGHT = h; // 后端摄像头/视频原始高度
    // ******************************************

    // 容器尺寸（Canvas 尺寸已同步为容器尺寸）
    const containerW = container.clientWidth;
    const containerH = container.clientHeight;

    // 计算等比例缩放比例（object-contain 逻辑）
    const scaleX = containerW / VIDEO_ORIGIN_WIDTH;
    const scaleY = containerH / VIDEO_ORIGIN_HEIGHT;
    const scale = Math.min(scaleX, scaleY); // 取最小比例，避免超出容器

    // 计算居中偏移量（视频流居中后，左右/上下黑边的偏移）
    const videoDisplayW = VIDEO_ORIGIN_WIDTH * scale; // 视频显示宽度
    const videoDisplayH = VIDEO_ORIGIN_HEIGHT * scale; // 视频显示高度
    const offsetX = (containerW - videoDisplayW) / 2; // 水平偏移（左右居中）
    const offsetY = (containerH - videoDisplayH) / 2; // 垂直偏移（上下居中）

    // // 打印调试信息（确认参数正确）
    // console.log(`
    //     变换参数：
    //     - 原始分辨率：${VIDEO_ORIGIN_WIDTH}x${VIDEO_ORIGIN_HEIGHT}
    //     - 容器尺寸：${containerW}x${containerH}
    //     - 缩放比例：${scale.toFixed(2)}
    //     - 显示尺寸：${videoDisplayW.toFixed(0)}x${videoDisplayH.toFixed(0)}
    //     - 偏移量：(${offsetX.toFixed(0)}, ${offsetY.toFixed(0)})
    // `);

    return { scale, offsetX, offsetY };
}




// 同步 Canvas 尺寸到容器尺寸（关键：确保绘制区域与视频容器一致）
function syncCanvasSize() {
    // const container = document.querySelector('.aspect-video');
    // document.getElementById("videoFeed")
    const container = videoFeed.parentElement;
    if (drawCanvas && container) {
        // 设置 Canvas 实际尺寸（避免拉伸）
        drawCanvas.width = container.clientWidth;
        drawCanvas.height = container.clientHeight;
        // 设置 Canvas 样式尺寸（占满容器）
        drawCanvas.style.width = `${container.clientWidth}px`;
        drawCanvas.style.height = `${container.clientHeight}px`;
        console.log(`Canvas 尺寸同步完成：${container.clientWidth}x${container.clientHeight}`);
    }
}

// 核心绘制函数
function drawTrackedTargets(trackedTargets, w, h) {
    // 1. 基础校验
    if (!draw_ctx) {
        console.error("绘制上下文未初始化！请先调用 initDrawCanvas()");
        return;
    }
    if (!trackedTargets || trackedTargets.length === 0) {
        // 无结果时清空画布
        draw_ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
        console.log("无跟踪结果可绘制");
        return;
    }
    // 2. 清空上一帧（避免残留）
    draw_ctx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    // 3. 遍历绘制每个目标
    trackedTargets.forEach((target, targetIndex) => {
        try {
            const { text, bbox, trackId, isOverLimit, similarity, userLabel } = target;
            console.log(`目标${targetIndex} - 原始数据：`, JSON.stringify({ text, bbox, isOverLimit }));
            const validPoints = [];
            let isBboxValid = true;

            // 先校验 bbox 本身是否为有效数组
            if (!Array.isArray(bbox) || bbox.length < 3) {
              console.warn(`目标${targetIndex} - bbox 无效：必须是≥3个顶点的数组`, bbox);
              isBboxValid = false;
            } else {
              // 遍历所有顶点（不再硬限制4个，适配任意长度）
              for (let i = 0; i < bbox.length; i++) {
                const point = bbox[i];
                // 校验点的格式（必须是长度为2的数组）
                if (!Array.isArray(point) || point.length !== 2) {
                  console.warn(`目标${targetIndex} - 第${i+1}个点格式无效：`, point);
                  isBboxValid = false;
                  break;
                }
                // 转换为数字并校验有效性（非NaN、非负、有限值）
                const x = Number(point[0]);
                const y = Number(point[1]);
                if (isNaN(x) || isNaN(y) || x < 0 || y < 0 || !isFinite(x) || !isFinite(y)) {
                  console.warn(`目标${targetIndex} - 第${i+1}个点坐标无效：`, point);
                  isBboxValid = false;
                  break;
                }
                validPoints.push([x, y]);
              }
            }

            // 额外校验：有效顶点数必须≥3（否则无法构成多边形）
            if (isBboxValid && validPoints.length < 3) {
              console.warn(`目标${targetIndex} - 有效顶点数不足3个：`, validPoints.length);
              isBboxValid = false;
            }

            if (!isBboxValid) return;

            // 4.2 应用坐标变换（原始坐标 → 显示坐标，适配任意顶点数）
            const { scale, offsetX, offsetY } = getVideoTransform(w, h);
            const transformedPoints = validPoints.map(([x, y]) => [
              x * scale + offsetX, // 缩放+水平偏移（适配左黑边）
              y * scale + offsetY  // 缩放+垂直偏移（适配上黑边）
            ]);
            console.log(`目标${targetIndex} - 转换后坐标（${transformedPoints.length}个顶点）：`, JSON.stringify(transformedPoints));

            // 4.3 计算目标框关键参数
            const xCoords = transformedPoints.map(p => p[0]);
            const yCoords = transformedPoints.map(p => p[1]);
            const minX = Math.min(...xCoords);
            const maxX = Math.max(...xCoords);
            const minY = Math.min(...yCoords);
            const maxY = Math.max(...yCoords);
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const boxWidth = maxX - minX;
            const boxHeight = maxY - minY;
            const minBoxSize = 8; // 最小可见尺寸

            // 4.4 调整过小目标框（确保可见）
            let targetBbox = transformedPoints;
            if (boxWidth < minBoxSize || boxHeight < minBoxSize) {
                targetBbox = [
                    [centerX - minBoxSize/2, centerY - minBoxSize/2],
                    [centerX + minBoxSize/2, centerY - minBoxSize/2],
                    [centerX + minBoxSize/2, centerY + minBoxSize/2],
                    [centerX - minBoxSize/2, centerY + minBoxSize/2]
                ];
            }

            // 4.5 设置绘制样式（适配设计风格）
            const borderColor = isOverLimit ? '#ff0000' : '#00ff00'; // 超限红/正常绿
            draw_ctx.strokeStyle = borderColor;
            draw_ctx.lineWidth = 3; // 加粗边框
            draw_ctx.fillStyle = `${borderColor}20`; // 半透明填充（突出目标）

            // 4.6 绘制目标框（任意顶点数的闭合多边形）
            if (targetBbox && targetBbox.length >= 3) { // 至少3个顶点才是有效多边形
              draw_ctx.beginPath();
              
              // 1. 移动到第一个顶点（起点）
              draw_ctx.moveTo(targetBbox[0][0], targetBbox[0][1]);
              
              // 2. 循环遍历剩余所有顶点（自动适配任意顶点数）
              for (let i = 1; i < targetBbox.length; i++) {
                const [x, y] = targetBbox[i]; // 解构当前顶点的 x、y 坐标
                draw_ctx.lineTo(x, y); // 连接到当前顶点
              }
              
              // 3. 闭合路径（自动连接最后一个顶点到第一个顶点）
              draw_ctx.closePath();
              
              // 4. 绘制边框和填充（保持原样式）
              draw_ctx.stroke(); // 绘制多边形边框
              draw_ctx.fill();   // 填充多边形内部（如需透明填充，可先设置 fillStyle）
            } else {
              console.warn("无效的多边形：顶点数不足3个", targetBbox);
            }

            // 4.7 绘制主文本（目标上方居中）
            draw_ctx.font = "14px Arial, 'Microsoft YaHei'";
            const mainText = `${text || '未知目标'}`;
            const textWidth = draw_ctx.measureText(mainText).width;
            const textHeight = 20;
            // 文本背景（半透明彩色）
            const textBgX = centerX - textWidth / 2 - 4;
            const textBgY = minY - textHeight - 2; // 目标上方5px
            draw_ctx.fillStyle = `${borderColor}80`;
            draw_ctx.fillRect(textBgX, textBgY, textWidth + 8, textHeight);
            // 文本内容（白色居中）
            draw_ctx.fillStyle = '#ffffff';
            draw_ctx.textAlign = 'center';
            draw_ctx.textBaseline = 'middle';
            draw_ctx.fillText(mainText, centerX, textBgY + textHeight/2);

            // // 4.8 绘制相似度（目标下方居中）
            // draw_ctx.font = "12px Arial";
            // const validSim = typeof similarity === 'number' && !isNaN(similarity) ? similarity : 0.00;
            // const simText = `相似度: ${validSim.toFixed(2)}`;
            // const simWidth = draw_ctx.measureText(simText).width;
            // const simBgHeight = 16;
            // 位置适配：避免超出画布底部
            // let simBgY = maxY + 5;
            // if (simBgY + simBgHeight > drawCanvas.height) {
            //     simBgY = minY - simBgHeight - 5; // 移到目标上方
            // }
            // // 相似度背景+文本
            // draw_ctx.fillStyle = `${borderColor}80`;
            // draw_ctx.fillRect(centerX - simWidth/2 - 4, simBgY, simWidth + 8, simBgHeight);
            // draw_ctx.fillStyle = '#ffffff';
            // draw_ctx.fillText(simText, centerX, simBgY + simBgHeight/2);

            // // 4.9 绘制标签+状态（相似度下方/上方）
            // const labelText = `${userLabel || '无标签'} | ${isOverLimit ? '超限' : '正常'}`;
            // const labelWidth = draw_ctx.measureText(labelText).width;
            // const labelBgHeight = 16;
            // 位置适配
            // let labelBgY = simBgY + simBgHeight + 3;
            // if (labelBgY + labelBgHeight > drawCanvas.height || simBgY < textBgY) {
            //     labelBgY = textBgY - labelBgHeight - 3; // 移到主文本上方
            // }
            // // 标签背景+文本
            // draw_ctx.fillStyle = `${borderColor}80`;
            // draw_ctx.fillRect(centerX - labelWidth/2 - 4, labelBgY, labelWidth + 8, labelBgHeight);
            // draw_ctx.fillStyle = '#ffffff';
            // draw_ctx.fillText(labelText, centerX, labelBgY + labelBgHeight/2);

        } catch (err) {
            console.error(`目标${targetIndex}绘制失败：`, err, `目标数据：`, JSON.stringify(target));
        }
    });
}

// 监听窗口大小变化（同步 Canvas 尺寸）
window.addEventListener('resize', syncCanvasSize);

// 监听视频流加载完成（确保变换参数正确）
document.getElementById('videoFeed').addEventListener('loadedmetadata', function() {
    console.log("视频流加载完成，分辨率：", this.videoWidth, "x", this.videoHeight);
    syncCanvasSize(); // 重新同步尺寸（避免流加载后容器尺寸变化）
});

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initDrawCanvas();
});






/*-----------------------------------------数据监测区域管理-----------------------------------------------------------------------*/
// /**
//  * 1. 初始化检测精度趋势图（折线图）
//  */
// function initAccuracyChart() {
//   const ctx = document.getElementById("accuracyChart").getContext("2d");
//   accuracyChart = new Chart(ctx, {
//     type: "line",
//     data: {
//       labels: Array.from({ length: monitorData.maxHistoryLength }, (_, i) => `#${i+1}`), // 帧序号
//       datasets: [{
//         label: "平均置信度",
//         data: Array(monitorData.maxHistoryLength).fill(0), // 初始填充0
//         borderColor: "#10b981", // 工业绿（匹配 text-industrial-success）
//         backgroundColor: "rgba(16, 185, 129, 0.1)",
//         borderWidth: 2,
//         tension: 0.4,
//         fill: true,
//         pointRadius: 2,
//         pointBackgroundColor: "#10b981"
//       }]
//     },
//     options: {
//       responsive: true,
//       maintainAspectRatio: false,
//       scales: {
//         x: {
//           grid: { color: "rgba(255, 255, 255, 0.05)" },
//           ticks: { color: "#9ca3af", font: { size: 10 } }
//         },
//         y: {
//           grid: { color: "rgba(255, 255, 255, 0.05)" },
//           ticks: { color: "#9ca3af", font: { size: 10 }, stepSize: 0.1 },
//           min: 0,
//           max: 1.0,
//           title: {
//             display: true,
//             text: "置信度",
//             color: "#9ca3af",
//             font: { size: 10 }
//           }
//         }
//       },
//       plugins: {
//         legend: {
//           display: false // 隐藏图例（节省空间）
//         },
//         tooltip: {
//           backgroundColor: "rgba(17, 24, 39, 0.8)",
//           borderColor: "#374151",
//           borderWidth: 1,
//           titleFont: { size: 11 },
//           bodyFont: { size: 10 },
//           padding: 8
//         }
//       }
//     }
//   });
// }
// /**
//  * 1. 重新初始化检测精度趋势图（修复横坐标+纵坐标问题）
//  */
// function initAccuracyChart() {
//   const ctx = document.getElementById("accuracyChart").getContext("2d");
//   accuracyChart = new Chart(ctx, {
//     type: "line",
//     data: {
//       labels: [], // 初始为空，动态生成
//       datasets: [{
//         label: "平均置信度",
//         data: [], // 初始为空，动态添加
//         borderColor: "#10b981", // 工业绿
//         backgroundColor: "rgba(16, 185, 129, 0.1)",
//         borderWidth: 2,
//         tension: 0.4,
//         fill: true,
//         pointRadius: 2,
//         pointBackgroundColor: "#10b981",
//         pointHoverRadius: 3 //  hover 时放大点
//       }]
//     },
//     options: {
//       responsive: true,
//       maintainAspectRatio: false,
//       scales: {
//         x: {
//           grid: { color: "rgba(255, 255, 255, 0.05)" },
//           ticks: { color: "#9ca3af", font: { size: 10 } },
//           offset: false // 标签不偏移，避免遮挡
//         },
//         y: {
//           grid: { color: "rgba(255, 255, 255, 0.05)" },
//           ticks: { 
//             color: "#9ca3af", 
//             font: { size: 10 },
//             stepSize: 0.05, // 刻度间隔0.05（0.8、0.85、0.9...1.0）
//             callback: function(value) {
//               return value.toFixed(2); // 强制显示2位小数
//             }
//           },
//           min: 0.8, // 固定纵坐标最小值
//           max: 1.0, // 固定纵坐标最大值
//           title: {
//             display: true,
//             text: "平均置信度",
//             color: "#9ca3af",
//             font: { size: 10 }
//           }
//         }
//       },
//       plugins: {
//         legend: { display: false },
//         tooltip: {
//           backgroundColor: "rgba(17, 24, 39, 0.8)",
//           borderColor: "#374151",
//           borderWidth: 1,
//           titleFont: { size: 11 },
//           bodyFont: { size: 10 },
//           padding: 8,
//           callbacks: {
//             title: function(tooltipItems) {
//               return `帧 ${tooltipItems[0].dataIndex + 1}`; // 显示帧序号
//             }
//           }
//         }
//       },
//       animation: {
//         duration: 0 // 关闭动画，避免移动时卡顿
//       }
//     }
//   });
// }

// 配套修改：初始化图表时标签设为空（后续由update函数初始化1-30）
function initAccuracyChart() {
  const ctx = document.getElementById("accuracyChart").getContext("2d");
  accuracyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: ["未跟踪"], // 初始显示“未跟踪”
      datasets: [{
        label: "平均置信度",
        data: [0.8],
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        borderWidth: 2,
        tension: 0.4,
        fill: true,
        pointRadius: 2,
        pointBackgroundColor: "#3b82f6",
        pointHoverRadius: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: "rgba(0, 0, 0, 0.05)" },
          ticks: { color: "#6b7280", font: { size: 10 } },
          fixed: true // 固定横坐标，不滚动
        },
        y: {
          grid: { color: "rgba(0, 0, 0, 0.05)" },
          ticks: { 
            color: "#6b7280", 
            font: { size: 10 },
            stepSize: 0.05,
            callback: value => value.toFixed(2)
          },
          min: 0.8,
          max: 1.0,
          title: {
            display: true,
            text: "平均置信度",
            color: "#6b7280",
            font: { size: 10 }
          }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(255, 255, 255, 0.9)",
          borderColor: "#e5e7eb",
          borderWidth: 1,
          titleColor: "#374151",
          bodyColor: "#6b7280",
          titleFont: { size: 11 },
          bodyFont: { size: 10 },
          padding: 8
        }
      },
      animation: { duration: 0 }
    }
  });
}

// /**
//  * 修复：更新检测精度趋势图数据（完全同步队列的先进先出，显示最新30帧）
//  */
// function updateAccuracyChartData() {
//   const dataset = accuracyChart.data.datasets[0];
//   const labels = accuracyChart.data.labels;
//   const currentAvgConfidence = monitorData.confidenceHistory.length > 0
//     ? monitorData.confidenceHistory[monitorData.confidenceHistory.length - 1]
//     : 0.8; // 无数据时填充0.8（贴合纵坐标范围）

//   // 核心修复：图表数据同步队列逻辑（先进先出，保留最新30帧）
//   // 1. 新数据入队（添加到队尾）
//   dataset.data.push(currentAvgConfidence);
//   // 2. 标签同步入队（帧序号）
//   labels.push((labels.length + 1).toString());

//   // 3. 超限时，数据和标签同时出队（删除队首最旧数据）
//   if (dataset.data.length > monitorData.maxHistoryLength) {
//     dataset.data.shift(); // 数据出队（最旧帧）
//     labels.shift();       // 标签出队（对应最旧帧序号）
//   }

//   // 首次开始跟踪：移除初始的「未跟踪」标签和数据
//   if (labels.length > 1 && labels[0] === "未跟踪") {
//     labels.shift();
//     dataset.data.shift();
//   }

//   // 强制更新图表（无动画）
//   accuracyChart.update("none");
// }

/**
 * 简化版：横坐标固定1-30，新数据覆盖最旧数据
 */
function updateAccuracyChartData() {

  const dataset = accuracyChart.data.datasets[0];
  const labels = accuracyChart.data.labels;
  const currentAvgConfidence = monitorData.confidenceHistory.length > 0
    ? monitorData.confidenceHistory[monitorData.confidenceHistory.length - 1]
    : 0.8;

  // 1. 初始化标签：固定1-30（仅首次执行）
  if (labels.length === 0 || labels[0] === "未跟踪") {
    labels.splice(0, labels.length); // 清空原有标签
    for (let i = 1; i <= monitorData.maxHistoryLength; i++) {
      labels.push(i.toString()); // 固定添加1-30
    }
    dataset.data.splice(0, dataset.data.length); // 清空原有数据
    dataset.data = Array(monitorData.maxHistoryLength).fill(0.8); // 初始填充0.8
  }

  // 2. 队列逻辑：新数据入队，超限时删除最旧数据（标签固定，只更数据）
  dataset.data.push(currentAvgConfidence);
  if (dataset.data.length > monitorData.maxHistoryLength) {
    dataset.data.shift();
  }

  // 强制更新图表（无动画）
  accuracyChart.update("none");
}






/**
 * 2. 初始化置信度阈值仪表盘（环形图）
 */
function initConfidenceGauge() {
  const ctx = document.getElementById("confidenceGauge").getContext("2d");
  confidenceGauge = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["阈值", "剩余"],
      datasets: [{
        data: [monitorData.confidenceThreshold * 100, 100 - (monitorData.confidenceThreshold * 100)],
        backgroundColor: [
          "#3b82f6", // 工业蓝（匹配主题）
          "rgba(75, 85, 99, 0.3)" // 灰色背景（透明）
        ],
        borderColor: "transparent",
        borderWidth: 0,
        cutout: "70%" // 环形宽度（70%空心）
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false }
      },
      circumference: 180, // 半圆仪表盘
      rotation: 270 // 旋转方向（从下往上）
    }
  });
}

/**
 * 3. 初始化跟踪稳定性仪表盘（环形图）
 */
function initStabilityGauge() {
  const ctx = document.getElementById("stabilityGauge").getContext("2d");
  stabilityGauge = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["稳定性", "剩余"],
      datasets: [{
        data: [monitorData.stability, 100 - monitorData.stability],
        backgroundColor: [
          "#10b981", // 工业绿（匹配 text-industrial-success）
          "rgba(75, 85, 99, 0.3)"
        ],
        borderColor: "transparent",
        borderWidth: 0,
        cutout: "70%"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false }
      },
      circumference: 180,
      rotation: 270
    }
  });
}

/**
 * 4. 更新监测数据（核心函数：对接业务逻辑）
 * @param {Array} trackedTargets - 跟踪目标列表（从后端获取的结果）
 * @param {Number} inferTime - 单次推理耗时（ms）
 */
function updateMonitorSource(trackedTargets = [], spottingNums = 0) {
  // 4.1 更新基础数据
  monitorData.targetCount = spottingNums; // 检测目标数 = 跟踪目标数（可根据实际业务调整）
  monitorData.trackCount = trackedTargets.filter(t => t.trackId).length; // 有效跟踪目标数（有trackId）
  const inferTime = 33
  monitorData.inferTime = Math.round(inferTime); // 推理耗时（取整）

  // 4.2 更新告警次数（累计超限目标数）
  const currentOverlimit = trackedTargets.filter(t => t.isOverLimit).length;
  if (currentOverlimit > 0) {
    monitorData.alarmCount += currentOverlimit; // 累计告警（可改为单次增量）
  }

  // 4.3 更新置信度历史（计算当前帧平均置信度）
  const totalConfidence = trackedTargets.reduce((sum, t) => sum + t.confidence, 0);
  const avgConfidence = trackedTargets.length > 0 
    ? Math.max(0.8, parseFloat((totalConfidence / trackedTargets.length).toFixed(2))) // 低于0.8按0.8显示（贴合纵坐标范围）
    : 0.8; // 无数据时填充0.8（避免图表底部空荡）

  // const totalConfidence = trackedTargets.reduce((sum, t) => sum + t.confidence, 0);
  // const avgConfidence = trackedTargets.length > 0 ? (totalConfidence / trackedTargets.length).toFixed(2) : 0;
  // 新增当前帧数据，移除最旧数据
  monitorData.confidenceHistory.push(parseFloat(avgConfidence));
  if (monitorData.confidenceHistory.length > monitorData.maxHistoryLength) {
    monitorData.confidenceHistory.shift();
  }

  // 计算跟踪稳定性
  if (globalState.currentTemplate && globalState.currentTemplate.content) {
    const templateTargetTotal = globalState.currentTemplate.content.length; // 模板中定义的目标总数（基准值）
    if (templateTargetTotal > 0) {
      // 计算稳定性（保留1位小数，避免波动过大）
      monitorData.stability = Math.round((monitorData.trackCount / templateTargetTotal) * 1000) / 10;
    } else {
      monitorData.stability = 0; // 模板无目标时，稳定性为0
    }
  } else {
    monitorData.stability = 0; // 未加载模板时，稳定性为0
  }
  // monitorData.stability = Math.max(85, Math.min(98, monitorData.stability + (Math.random() - 0.5) * 2));
}

/**
 * 5. 实时更新DOM和图表显示
 */
function updateMonitorData() {
  if (!isTrackActive || !isStreamActive) return;
  // 5.1 更新关键指标卡片
  updateTargetCountCard();
  updateTrackCountCard();
  updateInferTimeCard();
  updateAlarmCountCard();

  // 5.2 更新趋势图和仪表盘
  updateAccuracyChartData();
  updateConfidenceGaugeData();
  updateStabilityGaugeData();
}

/**
 * 辅助函数：更新检测目标数卡片（含变化率）
 */
function updateTargetCountCard() {
  const countEl = document.getElementById("targetCount");
  const changeEl = countEl.nextElementSibling;

  // 更新数值
  countEl.textContent = monitorData.targetCount;

  // 计算变化率
  const diff = monitorData.targetCount - monitorData.lastTargetCount;
  const changeRate = monitorData.lastTargetCount > 0 ? Math.round((diff / monitorData.lastTargetCount) * 100) : 0;

  // 更新变化率样式和文本
  if (diff > 0) {
    changeEl.className = "text-xs text-industrial-success";
    changeEl.innerHTML = `<i class="fa fa-arrow-up"></i> ${changeRate}%`;
  } else if (diff < 0) {
    changeEl.className = "text-xs text-industrial-danger";
    changeEl.innerHTML = `<i class="fa fa-arrow-down"></i> ${Math.abs(changeRate)}%`;
  } else {
    changeEl.className = "text-xs text-gray-400";
    changeEl.innerHTML = `<i class="fa fa-minus"></i> 0%`;
  }

  // 记录当前值为下次对比用
  monitorData.lastTargetCount = monitorData.targetCount;
}

/**
 * 辅助函数：更新跟踪目标数卡片（含变化率）
 */
function updateTrackCountCard() {
  const countEl = document.getElementById("trackCount");
  const changeEl = countEl.nextElementSibling;

  countEl.textContent = monitorData.trackCount;

  const diff = monitorData.trackCount - monitorData.lastTrackCount;
  const changeRate = monitorData.lastTrackCount > 0 ? Math.round((diff / monitorData.lastTrackCount) * 100) : 0;

  if (diff > 0) {
    changeEl.className = "text-xs text-industrial-success";
    changeEl.innerHTML = `<i class="fa fa-arrow-up"></i> ${changeRate}%`;
  } else if (diff < 0) {
    changeEl.className = "text-xs text-industrial-danger";
    changeEl.innerHTML = `<i class="fa fa-arrow-down"></i> ${Math.abs(changeRate)}%`;
  } else {
    changeEl.className = "text-xs text-gray-400";
    changeEl.innerHTML = `<i class="fa fa-minus"></i> 0%`;
  }

  monitorData.lastTrackCount = monitorData.trackCount;
}

/**
 * 辅助函数：更新推理耗时卡片
 */
function updateInferTimeCard() {
  const timeEl = document.getElementById("inferTime");
  timeEl.textContent = monitorData.inferTime;

  // 耗时颜色提示：<50ms绿色，50-100ms黄色，>100ms红色
  if (monitorData.inferTime < 50) {
    timeEl.className = "text-2xl font-bold text-industrial-success";
  } else if (monitorData.inferTime < 100) {
    timeEl.className = "text-2xl font-bold text-industrial-warning";
  } else {
    timeEl.className = "text-2xl font-bold text-industrial-danger";
  }
}

/**
 * 辅助函数：更新告警次数卡片
 */
function updateAlarmCountCard() {
  const alarmEl = document.getElementById("alarmCount");
  alarmEl.textContent = monitorData.alarmCount;

  // 告警次数>0时显示闪烁动画（可选增强视觉）
  if (monitorData.alarmCount > 0) {
    alarmEl.classList.add("animate-pulse");
  } else {
    alarmEl.classList.remove("animate-pulse");
  }
}

// /**
//  * 辅助函数：更新检测精度趋势图数据
//  */
// function updateAccuracyChartData() {
//   accuracyChart.data.datasets[0].data = monitorData.confidenceHistory;
//   accuracyChart.update();
// }

/**
 * 辅助函数：更新置信度阈值仪表盘数据
 */
// function updateConfidenceGaugeData() {
//   confidenceGauge.data.datasets[0].data = [
//     monitorData.confidenceThreshold * 100,
//     100 - (monitorData.confidenceThreshold * 100)
//   ];
//   document.getElementById("confidenceValue").textContent = monitorData.confidenceThreshold.toFixed(2);
//   confidenceGauge.update();
// }
/**
 * 更新置信度仪表盘数据（改为显示当前跟踪目标的平均置信度）
 */
function updateConfidenceGaugeData() {
  // 🔴 核心：计算当前跟踪目标的平均置信度
  let avgConfidence = 0;
  if (monitorData.targetCount > 0) { // 有检测目标时才计算
    // 从置信度历史中取最新值（即当前帧的平均置信度），或重新计算
    avgConfidence = monitorData.confidenceHistory.length > 0 
      ? monitorData.confidenceHistory[monitorData.confidenceHistory.length - 1] // 复用已计算的平均置信度
      : 0;
  }

  // 🔴 更新仪表盘数据（平均置信度转为百分比）
  confidenceGauge.data.datasets[0].data = [
    avgConfidence * 100, // 平均置信度（0-1 → 0-100）
    100 - (avgConfidence * 100) // 剩余部分（透明背景）
  ];

  // 🔴 更新显示值（保留2位小数）
  const confidenceEl = document.getElementById("confidenceValue");
  confidenceEl.textContent = avgConfidence.toFixed(2);

  // 🔴 平均置信度颜色提示：>0.85绿色，0.7-0.85黄色，<0.7红色（匹配业务常用阈值）
  if (avgConfidence > 0.85) {
    confidenceEl.className = "text-lg font-bold mt-2 text-industrial-success";
  } else if (avgConfidence > 0.7) {
    confidenceEl.className = "text-lg font-bold mt-2 text-industrial-warning";
  } else {
    confidenceEl.className = "text-lg font-bold mt-2 text-industrial-danger";
  }

  confidenceGauge.update();
}



/**
 * 辅助函数：更新跟踪稳定性仪表盘数据
 */
function updateStabilityGaugeData() {
  stabilityGauge.data.datasets[0].data = [monitorData.stability, 100 - monitorData.stability];
  const stabilityEl = document.getElementById("stabilityValue");
  stabilityEl.textContent = `${Math.round(monitorData.stability)}%`;

  // 稳定性颜色：>90%绿色，80-90%黄色，<80%红色
  if (monitorData.stability > 90) {
    stabilityEl.className = "text-lg font-bold mt-2 text-industrial-success";
  } else if (monitorData.stability > 80) {
    stabilityEl.className = "text-lg font-bold mt-2 text-industrial-warning";
  } else {
    stabilityEl.className = "text-lg font-bold mt-2 text-industrial-danger";
  }

  stabilityGauge.update();
}

/**
 * 对外暴露：更新监测数据源（对接你的业务逻辑）
 * @param {Array} trackedTargets - 后端返回的跟踪目标列表
 * @param {Number} inferTime - 单次推理耗时（ms）
 * @param {Number} confidenceThreshold - 置信度阈值（可选，默认0.85）
 */
function setMonitorSource(trackedTargets, spottingNums) {
  updateMonitorSource(trackedTargets, spottingNums);
  // if (confidenceThreshold !== undefined) {
  //   monitorData.confidenceThreshold = confidenceThreshold;
  // }
}