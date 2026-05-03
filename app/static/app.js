const ocrForm = document.getElementById("ocrForm");
const imageInput = document.getElementById("imageInput");
const textInput = document.getElementById("textInput");
const ocrStatus = document.getElementById("ocrStatus");
const detectBtn = document.getElementById("detectBtn");
const clearBtn = document.getElementById("clearBtn");

const resultCard = document.getElementById("resultCard");
const verdictEl = document.getElementById("verdict");
const percentEl = document.getElementById("percent");
const resultIdEl = document.getElementById("resultId");
const barFill = document.getElementById("barFill");
const historyEl = document.getElementById("history");

function setLoading(element, text) {
    element.textContent = text;
}

function showError(message) {
    alert(message);
}

ocrForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!imageInput.files.length) {
        showError("Выберите изображение.");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    setLoading(ocrStatus, "Распознаём изображение...");

    try {
        const response = await fetch("/api/ocr", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Ошибка OCR");
        }

        textInput.value = data.text || "";
        setLoading(ocrStatus, "Готово. Проверьте текст и исправьте OCR-ошибки.");
    } catch (error) {
        setLoading(ocrStatus, "");
        showError(error.message);
    }
});

detectBtn.addEventListener("click", async () => {
    const text = textInput.value.trim();

    if (text.length < 20) {
        showError("Введите или распознайте текст длиной минимум 20 символов.");
        return;
    }

    detectBtn.disabled = true;
    detectBtn.textContent = "Проверяем...";

    try {
        const response = await fetch("/api/detect", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({text}),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Ошибка детекции");
        }

        renderResult(data);
        await loadHistory();
    } catch (error) {
        showError(error.message);
    } finally {
        detectBtn.disabled = false;
        detectBtn.textContent = "Проверить на AI-генерацию";
    }
});

clearBtn.addEventListener("click", () => {
    textInput.value = "";
    resultCard.classList.add("hidden");
});

function renderResult(data) {
    const isAI = data.verdict === "AI_GENERATED";
    verdictEl.textContent = isAI ? "Вероятно, AI-сгенерированный текст" : "Вероятно, текст написан человеком";
    percentEl.textContent = `${data.ai_percent.toFixed(2)}%`;
    resultIdEl.textContent = `#${data.id}`;
    barFill.style.width = `${Math.min(100, Math.max(0, data.ai_percent))}%`;
    resultCard.classList.remove("hidden");
}

async function loadHistory() {
    const response = await fetch("/api/results?limit=10");
    const items = await response.json();

    if (!items.length) {
        historyEl.innerHTML = `<p class="muted">Пока нет сохранённых результатов.</p>`;
        return;
    }

    historyEl.innerHTML = items.map(item => {
        const shortText = item.text.length > 160 ? item.text.slice(0, 160) + "..." : item.text;
        const verdict = item.verdict === "AI_GENERATED" ? "AI" : "Human";
        return `
            <article class="history-item">
                <div>
                    <strong>#${item.id} · ${verdict} · ${item.ai_percent.toFixed(2)}%</strong>
                    <p>${escapeHtml(shortText)}</p>
                </div>
                <time>${new Date(item.created_at).toLocaleString("ru-RU")}</time>
            </article>
        `;
    }).join("");
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

loadHistory();
