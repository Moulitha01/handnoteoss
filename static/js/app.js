// ============================================================
// HandNote OSS
// Frontend JavaScript
// ============================================================


// ============================================================
// ELEMENTS
// ============================================================

const imageInput =
    document.getElementById("imageInput");

const uploadArea =
    document.getElementById("uploadArea");

const uploadPlaceholder =
    document.getElementById("uploadPlaceholder");

const previewContainer =
    document.getElementById("previewContainer");

const imagePreview =
    document.getElementById("imagePreview");

const removeImage =
    document.getElementById("removeImage");

const convertButton =
    document.getElementById("convertButton");

const processingMessage =
    document.getElementById("processingMessage");

const recognizedText =
    document.getElementById("recognizedText");

const lineCount =
    document.getElementById("lineCount");

const noteTitle =
    document.getElementById("noteTitle");

const noteTags =
    document.getElementById("noteTags");

const copyButton =
    document.getElementById("copyButton");

const saveButton =
    document.getElementById("saveButton");

const searchInput =
    document.getElementById("searchInput");

const notesContainer =
    document.getElementById("notesContainer");

const noteCount =
    document.getElementById("noteCount");

const ocrEngine =
    document.getElementById("ocrEngine");

const debugCrops =
    document.getElementById("debugCrops");

const engineLabel =
    document.getElementById("engineLabel");


// ============================================================
// STATE
// ============================================================

let selectedFile = null;

let currentNoteId = null;


// ============================================================
// OCR ENGINE UI
// ============================================================

function updateOcrEngineUi() {

    const tamilMode =
        ocrEngine.value === "tamil";


    const buttonText =
        convertButton.querySelector(
            "span:first-child"
        );


    if (buttonText) {

        buttonText.textContent =
            tamilMode
                ? "Convert with Tamil OCR"
                : "Convert with TrOCR";

    }


    if (engineLabel) {

        engineLabel.textContent =
            tamilMode
                ? "Tamil OCR / தமிழ்"
                : "TrOCR / English";

    }


    if (!selectedFile) {

        processingMessage.textContent =
            tamilMode
                ? "Choose a Tamil text image to begin."
                : "Choose a handwritten note to begin.";

    }

}


ocrEngine.addEventListener(
    "change",
    updateOcrEngineUi
);


// ============================================================
// CHOOSE IMAGE
// ============================================================

uploadPlaceholder.addEventListener(
    "click",
    () => {

        imageInput.click();

    }
);


// ============================================================
// FILE INPUT
// ============================================================

imageInput.addEventListener(
    "change",
    () => {

        const file =
            imageInput.files[0];


        if (!file) {
            return;
        }


        selectImage(file);

    }
);


// ============================================================
// SELECT IMAGE
// ============================================================

function selectImage(file) {

    if (
        !file.type.startsWith("image/")
    ) {

        alert(
            "Please select an image file."
        );

        return;

    }


    selectedFile =
        file;


    const reader =
        new FileReader();


    reader.onload =
        (event) => {

            imagePreview.src =
                event.target.result;


            uploadPlaceholder
                .classList
                .add("hidden");


            previewContainer
                .classList
                .remove("hidden");


            processingMessage.textContent =
                ocrEngine.value === "tamil"
                    ? "Image ready. Click Convert with Tamil OCR."
                    : "Image ready. Click Convert with TrOCR.";

        };


    reader.readAsDataURL(
        file
    );

}


// ============================================================
// REMOVE IMAGE
// ============================================================

removeImage.addEventListener(
    "click",
    (event) => {

        event.stopPropagation();


        selectedFile =
            null;


        imageInput.value =
            "";


        imagePreview.src =
            "";


        previewContainer
            .classList
            .add("hidden");


        uploadPlaceholder
            .classList
            .remove("hidden");


        updateOcrEngineUi();

    }
);


// ============================================================
// DRAG OVER
// ============================================================

uploadArea.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

    }
);


// ============================================================
// DROP IMAGE
// ============================================================

uploadArea.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();


        const files =
            event.dataTransfer.files;


        if (
            !files ||
            files.length === 0
        ) {

            return;

        }


        selectImage(
            files[0]
        );

    }
);


// ============================================================
// CONVERT / OCR
// ============================================================

convertButton.addEventListener(
    "click",
    async () => {

        if (!selectedFile) {

            alert(
                "Choose a handwritten image first."
            );

            return;

        }


        const engine =
            ocrEngine.value;


        // ----------------------------------------------------
        // FORM DATA
        // ----------------------------------------------------

        const formData =
            new FormData();


        formData.append(
            "image",
            selectedFile
        );


        formData.append(
            "engine",
            engine
        );


        // Keep OCR output raw.
        formData.append(
            "llm",
            "0"
        );


        formData.append(
            "debug",
            debugCrops &&
            debugCrops.checked
                ? "1"
                : "0"
        );


        // ----------------------------------------------------
        // LOADING STATE
        // ----------------------------------------------------

        processingMessage.textContent =
            engine === "tamil"
                ? "Running local Tamil OCR..."
                : "Preparing page, finding lines, and running local TrOCR...";


        convertButton.disabled =
            true;


        const buttonText =
            convertButton.querySelector(
                "span:first-child"
            );


        if (buttonText) {

            buttonText.textContent =
                engine === "tamil"
                    ? "Reading Tamil..."
                    : "Reading handwriting...";

        }


        // ----------------------------------------------------
        // SEND IMAGE
        // ----------------------------------------------------

        try {

            const response =
                await fetch(
                    "/api/upload",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            let result;


            try {

                result =
                    await response.json();

            }
            catch {

                throw new Error(
                    "Server returned an invalid response."
                );

            }


            if (!response.ok) {

                throw new Error(
                    result.error ||
                    `OCR failed (${response.status})`
                );

            }


            console.log(
                "OCR response:",
                result
            );


            // ------------------------------------------------
            // NOTE ID
            // ------------------------------------------------

            currentNoteId =
                result.id ?? null;


            // ------------------------------------------------
            // OCR TEXT
            // ------------------------------------------------

            const text =
                result.text ??
                result.ocr_text ??
                result.content ??
                "";


            recognizedText.value =
                text;


            // ------------------------------------------------
            // TITLE
            // ------------------------------------------------

            if (result.title) {

                noteTitle.value =
                    result.title;

            }
            else {

                noteTitle.value =
                    selectedFile.name.replace(
                        /\.[^/.]+$/,
                        ""
                    );

            }


            // ------------------------------------------------
            // TAGS
            // ------------------------------------------------

            noteTags.value =
                result.tags || "";


            updateLineCount();


            // ------------------------------------------------
            // ENGINE LABEL
            // ------------------------------------------------

            if (engineLabel) {

                engineLabel.textContent =
                    engine === "tamil"
                        ? "Tamil OCR / தமிழ்"
                        : "TrOCR / English";

            }


            processingMessage.textContent =
                `Finished locally using ${result.engine || engine}.`;


            await loadNotes();

        }

        catch (error) {

            console.error(
                "OCR error:",
                error
            );


            processingMessage.textContent =
                `OCR failed: ${error.message}`;

        }

        finally {

            convertButton.disabled =
                false;


            if (buttonText) {

                buttonText.textContent =
                    ocrEngine.value === "tamil"
                        ? "Convert with Tamil OCR"
                        : "Convert with TrOCR";

            }

        }

    }
);


// ============================================================
// LINE COUNT
// ============================================================

recognizedText.addEventListener(
    "input",
    updateLineCount
);


function updateLineCount() {

    const text =
        recognizedText.value.trim();


    if (!text) {

        lineCount.textContent =
            "0 lines";

        return;

    }


    const lines =
        text
            .split(/\r?\n/)
            .filter(
                line =>
                    line.trim().length > 0
            );


    const count =
        lines.length;


    lineCount.textContent =
        `${count} ${count === 1 ? "line" : "lines"}`;

}


// ============================================================
// COPY TEXT
// ============================================================

copyButton.addEventListener(
    "click",
    async () => {

        const text =
            recognizedText.value;


        if (!text.trim()) {

            alert(
                "There is no text to copy."
            );

            return;

        }


        try {

            await navigator.clipboard.writeText(
                text
            );


            const oldText =
                copyButton.textContent;


            copyButton.textContent =
                "Copied ✓";


            setTimeout(
                () => {

                    copyButton.textContent =
                        oldText;

                },
                1200
            );

        }

        catch (error) {

            console.error(
                "Copy error:",
                error
            );


            alert(
                "Could not copy the text."
            );

        }

    }
);


// ============================================================
// SAVE NOTE
// ============================================================

saveButton.addEventListener(
    "click",
    async () => {

        if (!currentNoteId) {

            alert(
                "Convert or select a note first."
            );

            return;

        }


        const title =
            noteTitle.value.trim()
            || "Untitled note";


        const text =
            recognizedText.value;


        const tags =
            noteTags.value.trim();


        try {

            const response =
                await fetch(
                    `/api/notes/${currentNoteId}`,
                    {
                        method: "PUT",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            title: title,
                            text: text,
                            tags: tags
                        })
                    }
                );


            const result =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    result.error ||
                    "Could not save note."
                );

            }


            const oldHTML =
                saveButton.innerHTML;


            saveButton.innerHTML =
                "Saved ✓";


            setTimeout(
                () => {

                    saveButton.innerHTML =
                        oldHTML;

                },
                1200
            );


            await loadNotes();

        }

        catch (error) {

            console.error(
                "Save error:",
                error
            );


            alert(
                `Could not save note: ${error.message}`
            );

        }

    }
);


// ============================================================
// EXPORT
// ============================================================

document
    .querySelectorAll(
        ".export-buttons button"
    )
    .forEach(
        button => {

            button.addEventListener(
                "click",
                () => {

                    const format =
                        button.dataset.format;


                    if (!currentNoteId) {

                        alert(
                            "Convert or select a note first."
                        );

                        return;

                    }


                    window.location.href =
                        `/api/notes/${currentNoteId}/export/${format}`;

                }
            );

        }
    );


// ============================================================
// LOAD NOTES
// ============================================================

async function loadNotes(
    query = ""
) {

    try {

        const response =
            await fetch(
                `/api/notes?q=${encodeURIComponent(query)}`
            );


        if (!response.ok) {

            throw new Error(
                "Could not load notes."
            );

        }


        const notes =
            await response.json();


        if (!Array.isArray(notes)) {

            throw new Error(
                "Invalid notes response."
            );

        }


        renderNotes(
            notes
        );

    }

    catch (error) {

        console.error(
            "Load notes error:",
            error
        );


        notesContainer.innerHTML =
            `
            <div class="empty-state">
                Could not load saved notes.
            </div>
            `;

    }

}


// ============================================================
// RENDER NOTES
// ============================================================

function renderNotes(notes) {

    noteCount.textContent =
        notes.length;


    if (
        notes.length === 0
    ) {

        notesContainer.innerHTML =
            `
            <div class="empty-state">
                Saved notes will appear here.
            </div>
            `;


        return;

    }


    notesContainer.innerHTML =
        "";


    notes.forEach(
        note => {

            const item =
                document.createElement(
                    "div"
                );


            item.className =
                "saved-note-item";


            item.style.padding =
                "18px 4px";


            item.style.borderBottom =
                "1px solid #20364e";


            item.style.cursor =
                "pointer";


            // TITLE

            const title =
                document.createElement(
                    "div"
                );


            title.textContent =
                note.title ||
                "Untitled note";


            title.style.color =
                "#f2f6ff";


            title.style.fontWeight =
                "700";


            title.style.marginBottom =
                "7px";


            // PREVIEW

            const preview =
                document.createElement(
                    "div"
                );


            const previewText =
                note.text || "";


            preview.textContent =
                previewText.length > 140
                    ? previewText.slice(
                        0,
                        140
                    ) + "..."
                    : previewText;


            preview.style.color =
                "#7896bb";


            preview.style.fontSize =
                "12px";


            preview.style.lineHeight =
                "1.6";


            item.appendChild(
                title
            );


            item.appendChild(
                preview
            );


            // TAGS

            if (note.tags) {

                const tags =
                    document.createElement(
                        "div"
                    );


                tags.textContent =
                    note.tags;


                tags.style.color =
                    "#a9f45b";


                tags.style.fontFamily =
                    "monospace";


                tags.style.fontSize =
                    "9px";


                tags.style.marginTop =
                    "8px";


                item.appendChild(
                    tags
                );

            }


            item.addEventListener(
                "click",
                () => {

                    openNote(
                        note
                    );

                }
            );


            notesContainer.appendChild(
                item
            );

        }
    );

}


// ============================================================
// OPEN SAVED NOTE
// ============================================================

function openNote(note) {

    currentNoteId =
        note.id;


    noteTitle.value =
        note.title || "";


    noteTags.value =
        note.tags || "";


    recognizedText.value =
        note.text || "";


    updateLineCount();


    processingMessage.textContent =
        `Opened saved note #${note.id}.`;


    const workspace =
        document.querySelector(
            ".workspace"
        );


    if (workspace) {

        workspace.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }

}


// ============================================================
// SEARCH
// ============================================================

let searchTimer =
    null;


searchInput.addEventListener(
    "input",
    () => {

        clearTimeout(
            searchTimer
        );


        searchTimer =
            setTimeout(
                () => {

                    loadNotes(
                        searchInput.value.trim()
                    );

                },
                250
            );

    }
);


// ============================================================
// INITIALIZE
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        updateOcrEngineUi();

        updateLineCount();

        loadNotes();

    }
);