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
const saveGradeForm = document.getElementById("saveGradeForm");
const classSelect = document.getElementById("classSelect");
const studentSelect = document.getElementById("studentSelect");
const workTypeSelect = document.getElementById("workTypeSelect");
const workDateInput = document.getElementById("workDateInput");
const gradeSelect = document.getElementById("gradeSelect");
const descriptionInput = document.getElementById("descriptionInput");
const saveStatus = document.getElementById("saveStatus");

const accountName = document.getElementById("accountName");
const accountLogin = document.getElementById("accountLogin");
const journalClassSelect = document.getElementById("journalClassSelect");
const journalDaysSelect = document.getElementById("journalDaysSelect");
const journalEl = document.getElementById("journal");
const manualGradeForm = document.getElementById("manualGradeForm");
const manualStudentSelect = document.getElementById("manualStudentSelect");
const manualWorkTypeSelect = document.getElementById("manualWorkTypeSelect");
const manualWorkDateInput = document.getElementById("manualWorkDateInput");
const manualGradeSelect = document.getElementById("manualGradeSelect");
const manualDescriptionInput = document.getElementById("manualDescriptionInput");
const manualSaveStatus = document.getElementById("manualSaveStatus");
const essayDialog = document.getElementById("essayDialog");
const dialogBody = document.getElementById("dialogBody");
const closeDialogBtn = document.getElementById("closeDialogBtn");

let currentResult = null;
let classes = [];
let authConfig = {enabled: false};
const aiHighThreshold = Number(document.body.dataset.threshold || 0.73) * 100;

function authHeaders() {
    return {};
}

async function apiFetch(url, options = {}) {
    const headers = {...authHeaders(), ...(options.headers || {})};
    const response = await fetch(url, {...options, headers});
    if (response.status === 401 && authConfig.enabled) {
        window.location.href = "/login";
    }
    return response;
}

function setLoading(element, text) {
    element.textContent = text;
}

function showError(message) {
    alert(message);
}

function todayIso() {
    return new Date().toISOString().slice(0, 10);
}

async function initAuth() {
    const response = await fetch("/api/auth/config");
    authConfig = await response.json();

    if (!authConfig.enabled) {
        return;
    }

    const meResponse = await fetch("/api/me");
    if (meResponse.status === 401) {
        window.location.href = "/login";
        return;
    }
    const user = await meResponse.json();
    accountName.textContent = user.display_name || user.username;
    accountLogin.textContent = user.username;
}

document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", async () => {
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tab).classList.add("active");
        if (button.dataset.tab === "journalTab") {
            await loadJournal();
        }
    });
});

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
        const response = await apiFetch("/api/ocr", {
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
        const response = await apiFetch("/api/detect", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({text}),
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Ошибка детекции");
        }

        currentResult = data;
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
    currentResult = null;
    resultCard.classList.add("hidden");
});

classSelect.addEventListener("change", () => fillStudents(classSelect.value));
journalClassSelect.addEventListener("change", () => {
    fillManualStudents(journalClassSelect.value);
    loadJournal();
});
journalDaysSelect.addEventListener("change", loadJournal);
closeDialogBtn.addEventListener("click", () => essayDialog.close());

saveGradeForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!currentResult) {
        showError("Сначала проверьте сочинение.");
        return;
    }

    const payload = {
        class_id: Number(classSelect.value),
        student_id: Number(studentSelect.value),
        grade: Number(gradeSelect.value),
        work_date: workDateInput.value,
        work_type: workTypeSelect.value,
        description: descriptionInput.value.trim() || "Сочинение",
        detection_result_id: currentResult.id,
    };

    try {
        saveStatus.textContent = "Сохраняем оценку...";
        const response = await apiFetch("/api/grades", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Не удалось сохранить оценку.");
        }

        saveStatus.textContent = "Оценка сохранена в журнал.";
        await loadJournal();
    } catch (error) {
        saveStatus.textContent = "";
        showError(error.message);
    }
});

manualGradeForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = {
        class_id: Number(journalClassSelect.value),
        student_id: Number(manualStudentSelect.value),
        grade: Number(manualGradeSelect.value),
        work_date: manualWorkDateInput.value,
        work_type: manualWorkTypeSelect.value,
        description: manualDescriptionInput.value.trim() || workTypeLabel(manualWorkTypeSelect.value),
    };

    try {
        manualSaveStatus.textContent = "Добавляем оценку...";
        const response = await apiFetch("/api/grades", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Не удалось добавить оценку.");
        }

        manualSaveStatus.textContent = "Оценка добавлена.";
        manualDescriptionInput.value = "";
        await loadJournal();
    } catch (error) {
        manualSaveStatus.textContent = "";
        showError(error.message);
    }
});

function renderResult(data) {
    const isAI = data.verdict === "AI_GENERATED";
    verdictEl.textContent = isAI ? "Вероятно, AI-сгенерированный текст" : "Вероятно, текст написан человеком";
    percentEl.textContent = `${data.ai_percent.toFixed(2)}%`;
    resultIdEl.textContent = `#${data.id}`;
    barFill.style.width = `${Math.min(100, Math.max(0, data.ai_percent))}%`;
    resultCard.classList.remove("hidden");
}

async function loadClasses() {
    const response = await apiFetch("/api/classes");
    const items = await response.json();
    if (!response.ok) {
        throw new Error(items.detail || "Не удалось загрузить классы.");
    }

    classes = items;
    const options = classes.map((item) => `<option value="${item.id}">${escapeHtml(item.name)}</option>`).join("");
    classSelect.innerHTML = options;
    journalClassSelect.innerHTML = options;
    if (classes.length) {
        fillStudents(classes[0].id);
        fillManualStudents(classes[0].id);
    }
}

function fillStudents(classId) {
    const schoolClass = classes.find((item) => item.id === Number(classId));
    studentSelect.innerHTML = (schoolClass?.students || [])
        .map((student) => `<option value="${student.id}">${escapeHtml(student.full_name)}</option>`)
        .join("");
}

function fillManualStudents(classId) {
    const schoolClass = classes.find((item) => item.id === Number(classId));
    manualStudentSelect.innerHTML = (schoolClass?.students || [])
        .map((student) => `<option value="${student.id}">${escapeHtml(student.full_name)}</option>`)
        .join("");
}

async function loadHistory() {
    const response = await apiFetch("/api/results?limit=10");
    const items = await response.json();

    if (!items.length) {
        historyEl.innerHTML = `<p class="muted">Пока нет сохранённых результатов.</p>`;
        return;
    }

    historyEl.innerHTML = items.map((item) => {
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

async function loadJournal() {
    if (!journalClassSelect.value) {
        journalEl.innerHTML = `<p class="muted">Нет доступных классов.</p>`;
        return;
    }

    const response = await apiFetch(`/api/journal/${journalClassSelect.value}?days=${journalDaysSelect.value}`);
    const data = await response.json();
    if (!response.ok) {
        journalEl.innerHTML = `<p class="muted">${escapeHtml(data.detail || "Журнал недоступен.")}</p>`;
        return;
    }

    renderJournal(data);
}

function renderJournal(data) {
    const dates = data.dates;
    const header = dates.map((day) => `<th>${formatDate(day)}</th>`).join("");
    const rows = data.students.map((student) => {
        const cells = dates.map((day) => {
            const grades = student.grades.filter((grade) => grade.work_date === day);
            if (!grades.length) {
                return "<td></td>";
            }
            return `<td>${grades.map(renderGradeBadge).join("")}</td>`;
        }).join("");
        const avg = student.average_grade === null ? "" : formatAverage(student.average_grade);
        return `
            <tr>
                <th class="student-name">${escapeHtml(student.full_name)}</th>
                ${cells}
                <td class="average">${avg}</td>
            </tr>
        `;
    }).join("");

    journalEl.innerHTML = `
        <div class="journal-scroll">
            <table>
                <thead>
                    <tr>
                        <th class="student-name">Ученик</th>
                        ${header}
                        <th>Средний балл</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;

    journalEl.querySelectorAll("[data-result-id]").forEach((button) => {
        button.addEventListener("click", () => openEssay(button.dataset.resultId));
    });
}

function renderGradeBadge(grade) {
    const risk = riskClass(grade);
    const title = [
        workTypeLabel(grade.work_type),
        grade.description,
        grade.ai_percent === null ? null : `AI: ${grade.ai_percent.toFixed(2)}%`,
    ].filter(Boolean).join(" · ");
    const attrs = grade.detection_result_id ? `data-result-id="${grade.detection_result_id}"` : "";
    return `
        <button class="grade-badge ${risk}" title="${escapeHtml(title)}" ${attrs}>
            ${grade.grade}
        </button>
    `;
}

function riskClass(grade) {
    if (grade.work_type !== "essay" || grade.ai_percent === null) {
        return "risk-neutral";
    }
    if (grade.ai_percent >= aiHighThreshold) {
        return "risk-high";
    }
    if (grade.ai_percent >= 40) {
        return "risk-medium";
    }
    return "risk-low";
}

async function openEssay(resultId) {
    const response = await apiFetch(`/api/results/${resultId}`);
    const data = await response.json();
    if (!response.ok) {
        showError(data.detail || "Не удалось открыть сочинение.");
        return;
    }

    const verdict = data.verdict === "AI_GENERATED" ? "Вероятно AI-сгенерированный текст" : "Вероятно написано человеком";
    dialogBody.innerHTML = `
        <h2>Сочинение #${data.id}</h2>
        <div class="result-grid modal-result">
            <div><p class="muted">Вердикт</p><h3>${verdict}</h3></div>
            <div><p class="muted">AI-генерация</p><h3>${data.ai_percent.toFixed(2)}%</h3></div>
            <div><p class="muted">Дата проверки</p><h3>${new Date(data.created_at).toLocaleString("ru-RU")}</h3></div>
        </div>
        <h3>Оцифровка</h3>
        <pre>${escapeHtml(data.text)}</pre>
    `;
    essayDialog.showModal();
}

function workTypeLabel(value) {
    return {
        essay: "Сочинение",
        lesson_answer: "Ответ на уроке",
        dictation: "Диктант",
        test: "Контрольная",
    }[value] || value;
}

function formatDate(value) {
    return new Date(`${value}T00:00:00`).toLocaleDateString("ru-RU", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
    });
}

function formatAverage(value) {
    return Number(value).toLocaleString("ru-RU", {maximumFractionDigits: 2});
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

async function init() {
    workDateInput.value = todayIso();
    manualWorkDateInput.value = todayIso();
    await initAuth();
    await loadClasses();
    await loadHistory();
    await loadJournal();
}

init().catch((error) => showError(error.message));
