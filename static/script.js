
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear");
const predictBtn = document.getElementById("predict");
const resultDiv = document.getElementById("result");
const themeToggleBtn = document.getElementById("theme-toggle");
const imageUpload = document.getElementById("image-upload");
const uploadTrigger = document.getElementById("upload-trigger");

const currentTheme = localStorage.getItem("theme") || "light";
document.documentElement.setAttribute("data-theme", currentTheme);
themeToggleBtn.innerText = currentTheme === "dark" ? "☀️" : "🌙";

themeToggleBtn.addEventListener("click", () => {
  let theme = document.documentElement.getAttribute("data-theme");
  if (theme === "dark") {
    document.documentElement.setAttribute("data-theme", "light");
    localStorage.setItem("theme", "light");
    themeToggleBtn.innerText = "🌙";
  } else {
    document.documentElement.setAttribute("data-theme", "dark");
    localStorage.setItem("theme", "dark");
    themeToggleBtn.innerText = "☀️";
  }
});

ctx.lineWidth = 16;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.strokeStyle = "#000000";

let drawing = false;


function initCanvas() {
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}


initCanvas();

function startDrawing(e) {
  drawing = true;
  draw(e);
}

function stopDrawing() {
  drawing = false;
  ctx.beginPath();
}

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX || e.touches[0].clientX) - rect.left;
  const y = (e.clientY || e.touches[0].clientY) - rect.top;

  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("touchstart", (e) => {
  e.preventDefault();
  startDrawing(e);
});
canvas.addEventListener("touchend", stopDrawing);
canvas.addEventListener("touchmove", (e) => {
  e.preventDefault();
  draw(e);
});

clearBtn.addEventListener("click", () => {
  initCanvas();
  resultDiv.innerHTML = "Draw something to begin!";
  imageUpload.value = "";
});

uploadTrigger.addEventListener("click", () => {
  imageUpload.click();
});

function getAverageBrightness(img) {
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = img.width;
  tempCanvas.height = img.height;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.drawImage(img, 0, 0);
  const data = tempCtx.getImageData(0, 0, img.width, img.height).data;
  let sum = 0;
  const pixels = img.width * img.height;
  for (let i = 0; i < data.length; i += 4) {
    sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return sum / pixels;
}

imageUpload.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (event) {
    const img = new Image();
    img.onload = function () {
      const avgBrightness = getAverageBrightness(img);

      ctx.fillStyle = avgBrightness < 128 ? "#000000" : "#FFFFFF";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const scale = Math.min(
        canvas.width / img.width,
        canvas.height / img.height
      );
      const x = canvas.width / 2 - (img.width / 2) * scale;
      const y = canvas.height / 2 - (img.height / 2) * scale;

      ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
      resultDiv.innerText = "Image uploaded! Click 'Recognize' to analyze.";
    };
    img.src = event.target.result;
  };
  reader.readAsDataURL(file);
});

function canvasHasContent() {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3];
    if (alpha === 0) continue;
    const brightness =
      0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    if (brightness < 240) return true;
  }
  return false;
}


predictBtn.addEventListener("click", () => {
  if (!canvasHasContent()) {
    resultDiv.innerText = "Please draw or upload a digit first.";
    return;
  }

  resultDiv.innerText = "Analyzing...";
  const dataURL = canvas.toDataURL("image/png");

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataURL }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        resultDiv.innerText = data.error;
      } else {

        let probHtml = '<div class="prob-list">';
        for (let i = 0; i <= 9; i++) {
          const prob = data.all_probabilities[i];
          probHtml += `
                    <div class="prob-row">
                        <span class="prob-label">${i}</span>
                        <div class="prob-track">
                            <div class="prob-fill" style="width: ${prob}%;"></div>
                        </div>
                        <span class="prob-val">${prob}%</span>
                    </div>
                `;
        }
        probHtml += "</div>";

        resultDiv.innerHTML = `
                Predicted Digit: <span class="digit-res">${data.digit}</span>
                <span class="confidence-res">Confidence: ${data.confidence}</span>
                ${probHtml}
            `;
      }
    })
    .catch((err) => {
      console.error(err);
      resultDiv.innerText = "Error connecting to server.";
    });
});