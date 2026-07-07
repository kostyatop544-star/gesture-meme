import {
  FaceLandmarker,
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { SEED_MEMES } from "./seed_memes.js";

/* ------------------------------------------------------------------ */
/* Хранилище библиотеки мемов (localStorage: JSON с base64-картинками) */
/* ------------------------------------------------------------------ */

const STORAGE_KEY = "gesture_meme_library_v1";

const Library = {
  entries: [],

  load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      this.entries = raw ? JSON.parse(raw) : [];
    } catch {
      this.entries = [];
    }
  },

  save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.entries));
  },

  add({ name, imageDataUrl, referenceVector, threshold }) {
    const entry = {
      id: crypto.randomUUID(),
      name,
      imageDataUrl,
      referenceVector,
      threshold,
    };
    this.entries.push(entry);
    this.save();
    return entry;
  },

  remove(id) {
    this.entries = this.entries.filter((e) => e.id !== id);
    this.save();
    if (id.startsWith("seed-")) {
      const dismissed = getDismissedSeeds();
      if (!dismissed.includes(id)) {
        dismissed.push(id);
        localStorage.setItem(DISMISSED_SEEDS_KEY, JSON.stringify(dismissed));
      }
    }
  },
};

Library.load();

// Мерджим стартовый набор по id: новые записи из SEED_MEMES (например,
// добавленные позже) подтягиваются даже тем, кто уже открывал приложение
// раньше, но то, что пользователь осознанно удалил — не воскрешаем.
const DISMISSED_SEEDS_KEY = "gesture_meme_dismissed_seeds_v1";
function getDismissedSeeds() {
  try {
    return JSON.parse(localStorage.getItem(DISMISSED_SEEDS_KEY) || "[]");
  } catch {
    return [];
  }
}
const dismissed = new Set(getDismissedSeeds());
const existingIds = new Set(Library.entries.map((e) => e.id));
let addedAny = false;
for (const seed of SEED_MEMES) {
  if (!existingIds.has(seed.id) && !dismissed.has(seed.id)) {
    Library.entries.push({ ...seed });
    addedAny = true;
  }
}
if (addedAny) Library.save();

/* ------------------------------------------------------------------ */
/* Вектор признаков позы: та же идея, что в нативной версии,          */
/* но берём готовые blendshapes от MediaPipe вместо ручной геометрии  */
/* ------------------------------------------------------------------ */

const VECTOR_KEYS = [
  "jawOpen",
  "smile",
  "browRaise",
  "eyeOpenLeft",
  "eyeOpenRight",
  "tongueOut",
  "headTilt",
  "leftHandNearFace",
  "rightHandNearFace",
  "leftHandAboveHead",
  "rightHandAboveHead",
  "handsSpread",
];

function emptyVector() {
  const v = {};
  VECTOR_KEYS.forEach((k) => (v[k] = 0));
  return v;
}

function vectorDistance(a, b) {
  let sum = 0;
  for (const k of VECTOR_KEYS) {
    const d = (a[k] || 0) - (b[k] || 0);
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function blendshapeValue(categories, name) {
  const found = categories?.find((c) => c.categoryName === name);
  return found ? found.score : 0;
}

/* ------------------------------------------------------------------ */
/* MediaPipe: инициализация детекторов лица и рук                     */
/* ------------------------------------------------------------------ */

let faceLandmarker, handLandmarker;

async function initDetectors() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });

  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

/* ------------------------------------------------------------------ */
/* Камера                                                              */
/* ------------------------------------------------------------------ */

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const statusEl = document.getElementById("status");

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  video.play();
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
}

/* ------------------------------------------------------------------ */
/* Вычисление текущего вектора по кадру                                */
/* ------------------------------------------------------------------ */

function computeVector(faceResult, handResult) {
  const v = emptyVector();
  if (!faceResult || !faceResult.faceLandmarks?.length) return v;

  const landmarks = faceResult.faceLandmarks[0];
  const blend = faceResult.faceBlendshapes?.[0]?.categories;

  v.jawOpen = blendshapeValue(blend, "jawOpen");
  v.smile =
    (blendshapeValue(blend, "mouthSmileLeft") +
      blendshapeValue(blend, "mouthSmileRight")) /
    2;
  v.browRaise =
    (blendshapeValue(blend, "browInnerUp") +
      blendshapeValue(blend, "browOuterUpLeft") +
      blendshapeValue(blend, "browOuterUpRight")) /
    3;
  v.eyeOpenLeft = 1 - blendshapeValue(blend, "eyeBlinkLeft");
  v.eyeOpenRight = 1 - blendshapeValue(blend, "eyeBlinkRight");
  v.tongueOut = blendshapeValue(blend, "tongueOut");

  // Наклон головы — по углу между внешними уголками глаз
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  v.headTilt = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

  // Bounding box лица для нормализации положения рук
  const xs = landmarks.map((p) => p.x);
  const ys = landmarks.map((p) => p.y);
  const faceMinX = Math.min(...xs), faceMaxX = Math.max(...xs);
  const faceMinY = Math.min(...ys), faceMaxY = Math.max(...ys);
  const faceCx = (faceMinX + faceMaxX) / 2;
  const faceCy = (faceMinY + faceMaxY) / 2;
  const faceSize = Math.max(faceMaxX - faceMinX, faceMaxY - faceMinY) || 0.01;

  const hands = handResult?.landmarks || [];
  const handedness = handResult?.handedness || [];

  hands.forEach((hand, i) => {
    // MediaPipe отдаёт handedness с точки зрения камеры (уже зеркалировано под селфи)
    const label = handedness[i]?.[0]?.categoryName; // "Left" | "Right"
    const wrist = hand[0];
    const dist = Math.hypot(wrist.x - faceCx, wrist.y - faceCy);
    const near = Math.max(0, 1 - dist / (faceSize * 3));
    const above = wrist.y < faceMinY ? 1 : 0;

    if (label === "Left") {
      v.leftHandNearFace = Math.min(1, near);
      v.leftHandAboveHead = above;
    } else if (label === "Right") {
      v.rightHandNearFace = Math.min(1, near);
      v.rightHandAboveHead = above;
    }
  });

  if (hands.length === 2) {
    const d = Math.hypot(
      hands[0][0].x - hands[1][0].x,
      hands[0][0].y - hands[1][0].y
    );
    v.handsSpread = Math.min(1, d / (faceSize * 4));
  }

  return v;
}

/* ------------------------------------------------------------------ */
/* Подбор мема со сглаживанием по нескольким кадрам                    */
/* ------------------------------------------------------------------ */

const STABILITY_WINDOW = 6;
let recentMatches = [];

function bestMatch(vector) {
  let best = null;
  for (const entry of Library.entries) {
    const d = vectorDistance(vector, entry.referenceVector);
    if (d <= entry.threshold && (!best || d < best.distance)) {
      best = { entry, distance: d };
    }
  }

  recentMatches.push(best?.entry.id ?? null);
  if (recentMatches.length > STABILITY_WINDOW) recentMatches.shift();

  if (recentMatches.length < STABILITY_WINDOW) return best?.entry ?? null;

  const counts = {};
  recentMatches.forEach((id) => (counts[id] = (counts[id] || 0) + 1));
  const [stableId, count] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  if (stableId === "null" || count <= STABILITY_WINDOW / 2) return null;
  return Library.entries.find((e) => e.id === stableId) ?? null;
}

/* ------------------------------------------------------------------ */
/* Главный цикл детекции                                               */
/* ------------------------------------------------------------------ */

const memeImg = document.getElementById("memeImg");
const memeEmpty = document.getElementById("memeEmpty");
const matchTitle = document.getElementById("matchTitle");

let currentVector = emptyVector();
let currentMatchId = null;
let lastAddModeVector = null;

function renderMatch(entry) {
  if (entry?.id === currentMatchId) return;
  currentMatchId = entry?.id ?? null;

  if (entry) {
    memeImg.src = entry.imageDataUrl;
    memeImg.classList.remove("hidden");
    memeEmpty.classList.add("hidden");
    matchTitle.textContent = `${entry.name}.png`;
  } else {
    memeImg.classList.add("hidden");
    memeEmpty.classList.remove("hidden");
    matchTitle.textContent = "meme_detectado.png";
  }
}

function detectLoop() {
  const now = performance.now();

  if (video.readyState >= 2) {
    const faceResult = faceLandmarker.detectForVideo(video, now);
    const handResult = handLandmarker.detectForVideo(video, now);

    currentVector = computeVector(faceResult, handResult);
    lastAddModeVector = currentVector;

    const match = bestMatch(currentVector);
    renderMatch(match);

    drawOverlay(faceResult, handResult);
  }

  requestAnimationFrame(detectLoop);
}

function drawOverlay(faceResult, handResult) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  overlayCtx.fillStyle = "#39ff88";

  const drawPoints = (points, radius) => {
    for (const p of points) {
      overlayCtx.beginPath();
      overlayCtx.arc(p.x * overlay.width, p.y * overlay.height, radius, 0, Math.PI * 2);
      overlayCtx.fill();
    }
  };

  handResult?.landmarks?.forEach((hand) => drawPoints(hand, 2.5));
  // Точки лица рисуем реже, чтобы не забивать кадр
  const face = faceResult?.faceLandmarks?.[0];
  if (face) drawPoints(face.filter((_, i) => i % 6 === 0), 1.2);
}

/* ------------------------------------------------------------------ */
/* UI: модалка добавления мема                                         */
/* ------------------------------------------------------------------ */

const modalAdd = document.getElementById("modalAdd");
const btnAdd = document.getElementById("btnAdd");
const closeAdd = document.getElementById("closeAdd");
const fileInput = document.getElementById("fileInput");
const filePreview = document.getElementById("filePreview");
const nameInput = document.getElementById("nameInput");
const btnCapturePose = document.getElementById("btnCapturePose");
const poseStatus = document.getElementById("poseStatus");
const thresholdInput = document.getElementById("thresholdInput");
const thresholdVal = document.getElementById("thresholdVal");
const btnSaveMeme = document.getElementById("btnSaveMeme");

let pendingImageDataUrl = null;
let pendingVector = null;

function resizeImageFile(file, maxSize = 480) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, maxSize / Math.max(img.width, img.height));
        const canvas = document.createElement("canvas");
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.85));
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
}

function updateSaveButton() {
  btnSaveMeme.disabled = !(pendingImageDataUrl && pendingVector && nameInput.value.trim());
}

btnAdd.addEventListener("click", () => modalAdd.classList.remove("hidden"));
closeAdd.addEventListener("click", () => modalAdd.classList.add("hidden"));

fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  pendingImageDataUrl = await resizeImageFile(file);
  filePreview.src = pendingImageDataUrl;
  filePreview.classList.remove("hidden");
  updateSaveButton();
});

nameInput.addEventListener("input", updateSaveButton);

btnCapturePose.addEventListener("click", () => {
  pendingVector = { ...lastAddModeVector };
  poseStatus.textContent = "поза записана ✓";
  poseStatus.classList.add("ok");
  updateSaveButton();
});

thresholdInput.addEventListener("input", () => {
  thresholdVal.textContent = Number(thresholdInput.value).toFixed(2);
});

btnSaveMeme.addEventListener("click", () => {
  Library.add({
    name: nameInput.value.trim(),
    imageDataUrl: pendingImageDataUrl,
    referenceVector: pendingVector,
    threshold: Number(thresholdInput.value),
  });
  renderLibraryGrid();
  updateLibCount();

  // сброс формы
  pendingImageDataUrl = null;
  pendingVector = null;
  fileInput.value = "";
  nameInput.value = "";
  filePreview.classList.add("hidden");
  poseStatus.textContent = "поза не записана";
  poseStatus.classList.remove("ok");
  btnSaveMeme.disabled = true;
  modalAdd.classList.add("hidden");
});

/* ------------------------------------------------------------------ */
/* UI: модалка библиотеки                                              */
/* ------------------------------------------------------------------ */

const modalLibrary = document.getElementById("modalLibrary");
const btnLibrary = document.getElementById("btnLibrary");
const closeLibrary = document.getElementById("closeLibrary");
const libraryGrid = document.getElementById("libraryGrid");
const libraryEmptyHint = document.getElementById("libraryEmptyHint");
const libCount = document.getElementById("libCount");

btnLibrary.addEventListener("click", () => {
  renderLibraryGrid();
  modalLibrary.classList.remove("hidden");
});
closeLibrary.addEventListener("click", () => modalLibrary.classList.add("hidden"));

function renderLibraryGrid() {
  libraryGrid.innerHTML = "";
  libraryEmptyHint.classList.toggle("hidden", Library.entries.length > 0);

  Library.entries.forEach((entry) => {
    const item = document.createElement("div");
    item.className = "grid-item";
    item.innerHTML = `
      <img src="${entry.imageDataUrl}" alt="${entry.name}" />
      <div class="label">${entry.name}</div>
      <button class="del" title="удалить">×</button>
    `;
    item.querySelector(".del").addEventListener("click", () => {
      Library.remove(entry.id);
      renderLibraryGrid();
      updateLibCount();
    });
    libraryGrid.appendChild(item);
  });
}

function updateLibCount() {
  libCount.textContent = Library.entries.length;
}

/* ------------------------------------------------------------------ */
/* Запуск                                                               */
/* ------------------------------------------------------------------ */

async function main() {
  updateLibCount();
  try {
    statusEl.textContent = "загрузка моделей...";
    await initDetectors();
    statusEl.textContent = "запрос камеры...";
    await initCamera();
    statusEl.textContent = "● live";
    detectLoop();
  } catch (err) {
    console.error(err);
    statusEl.textContent = "ошибка: " + err.message;
  }
}

main();
