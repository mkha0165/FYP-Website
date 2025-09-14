/* upload.js
   Handles CSV upload + ML visualization calls (FastAPI backend).
*/

const ML_API_BASE_URL = window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "https://fyp-website-3.onrender.com";  // FastAPI service

document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("uploadForm");
    const fileInput = document.getElementById("csvFile");
    const resultContainer = document.getElementById("result");

    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        if (!fileInput.files.length) {
            alert("Please select a CSV file first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        try {
            const res = await fetch(`${ML_API_BASE_URL}/upload`, {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                throw new Error("Upload failed");
            }

            const data = await res.json();
            console.log("ML API response:", data);

            // Example: render results
            resultContainer.innerHTML = `
                <h3>Analysis Results</h3>
                <pre>${JSON.stringify(data, null, 2)}</pre>
            `;
        } catch (err) {
            console.error(err);
            resultContainer.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
        }
    });
});