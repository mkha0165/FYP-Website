/*
  Purpose: Handles frontend logic for login and signup forms.
  Features:
    - Form switching (login <-> signup).
    - Input validation and error messages.
    - Sends requests to backend (server.js) for signup/login.
    - Stores login state in localStorage for authentication.
*/

/* --- FORM UTILITIES --- */
// Display a form-wide message (success/error)
function setFormMessage(formElement, type, message) {
    const messageElement = formElement.querySelector(".form__message");
    messageElement.textContent = message;
    messageElement.classList.remove("form__message--success", "form__message--error");
    messageElement.classList.add(`form__message--${type}`);
}

// Add input-specific error
function setInputError(inputElement, message) {
    inputElement.classList.add("form__input--error");
    inputElement.parentElement.querySelector(".form__input-error-message").textContent = message;
}

// Clear error from input
function clearInputError(inputElement) {
    inputElement.classList.remove("form__input--error");
    inputElement.parentElement.querySelector(".form__input-error-message").textContent = "";
}

/* --- EVENT HANDLERS --- */
document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.querySelector('#login');
    const createAccountForm = document.querySelector('#createAccount');

    // Toggle from login -> signup
    document.querySelector("#linkCreateAccount").addEventListener("click", e => {
        e.preventDefault();
        loginForm.classList.add("form--hidden");
        createAccountForm.classList.remove("form--hidden");
    });

    // Toggle from signup -> login
    document.querySelector("#linkLogin").addEventListener("click", e => {
        e.preventDefault();
        loginForm.classList.remove("form--hidden");
        createAccountForm.classList.add("form--hidden");
    });

    // Login Form
    loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const identifier = document.querySelector("#loginUsername").value.trim();
    const password = document.querySelector("#loginPassword").value.trim();

    if (!identifier || !password) {
        setFormMessage(loginForm, "error", "Please fill in all fields");
        return;
    }

    try {
        // Send login request to backend
        const res = await fetch("http://localhost:5000/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ identifier, password })
        });

        const data = await res.json();

        if (res.ok && data.success) {
            setFormMessage(loginForm, "success", data.message);

             // Save login status in localStorage
            localStorage.setItem("loggedInUser", identifier);

            // Redirect to CSV upload page
            setTimeout(() => {
                window.location.href = "upload.html";
            }, 1000);
        } else {
            setFormMessage(loginForm, "error", data.message || "Login failed");
        }
    } catch (err) {
        console.error("Login error:", err);
        setFormMessage(loginForm, "error", "Server error. Please try again.");
    }
});


    // Real-time validation
    document.querySelectorAll(".form__input").forEach(inputElement => {
        inputElement.addEventListener("input", e => {
            clearInputError(inputElement);

            // Username validation
            if (e.target.id === "signupUsername") {
                if (e.target.value.length > 0 && e.target.value.length < 10) {
                    setInputError(inputElement, "Username must be at least 10 characters in length");
                }
            }

            // Email validation
            if (e.target.id === "signupEmail") {
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (e.target.value.length > 0 && !emailRegex.test(e.target.value)) {
                    setInputError(inputElement, "Please enter a valid email address");
                }
            }

            // Password validation
            if (e.target.id === "signupPassword") {
                const password = e.target.value;
                if (password.length > 0 && password.length < 8) {
                    setInputError(inputElement, "Password must be at least 8 characters long");
                } else if (password.length > 0 && !/\d/.test(password)) {
                    setInputError(inputElement, "Password must contain at least one number");
                }
            }

            // Confirm password validation
            if (e.target.id === "signupConfirmPassword") {
                const password = document.querySelector("#signupPassword").value;
                if (e.target.value.length > 0 && e.target.value !== password) {
                    setInputError(inputElement, "Passwords do not match");
                }
            }
        });
    });

    // Single signup handler
    createAccountForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Final validation
        const inputs = createAccountForm.querySelectorAll(".form__input");
        let hasErrors = false;
        inputs.forEach(input => {
            if (input.classList.contains("form__input--error") || input.value.trim() === "") {
                hasErrors = true;
            }
        });

        if (hasErrors) {
            setFormMessage(createAccountForm, "error", "Please fix the errors before submitting");
            return;
        }

        // Collect form values
        const username = document.querySelector("#signupUsername").value;
        const email = document.querySelector("#signupEmail").value;
        const password = document.querySelector("#signupPassword").value;

        try {
            // Send signup request
            const res = await fetch("http://localhost:5000/signup", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, email, password })
            });

            const data = await res.json();

            if (res.ok) {
                setFormMessage(createAccountForm, "success", data.message);
                setTimeout(() => {
                    createAccountForm.classList.add("form--hidden");
                    loginForm.classList.remove("form--hidden");
                }, 1200);
            } else {
                setFormMessage(createAccountForm, "error", data.error || "Signup failed");
            }
        } catch (err) {
            console.error(err);
            setFormMessage(createAccountForm, "error", "Server error. Please try again.");
        }
    });
});
